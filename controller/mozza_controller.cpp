#include <gst/gst.h>
#include <iostream>
#include <vector>
#include <string.h>

// Structure to hold scheduled alpha changes
struct AlphaChange {
    GstClockTime time;
    gdouble alpha;
};

// Global references to pipeline and changes for simplicity
static GstElement *pipeline = nullptr;
static GstElement *mozza = nullptr;
static std::vector<AlphaChange> changes;

// Callback for new pads in demuxer (unchanged)
void on_pad_added(GstElement *element, GstPad *pad, gpointer data) {
    GstElement *decoder = (GstElement *)data;
    GstPad *sinkpad = gst_element_get_static_pad(decoder, "sink");
    gst_pad_link(pad, sinkpad);
    gst_object_unref(sinkpad);
}

// Timeout callback to check if we should apply alpha changes
static gboolean update_alpha(gpointer user_data) {
    if (!pipeline) return G_SOURCE_CONTINUE;

    GstFormat fmt = GST_FORMAT_TIME;
    gint64 pos = 0;

    // Get current pipeline position
    if (gst_element_query_position(pipeline, fmt, &pos)) {
        // Apply any changes that are due at or before current position
        for (auto it = changes.begin(); it != changes.end();) {
            if (it->time <= (GstClockTime)pos) {
                g_object_set(mozza, "alpha", it->alpha, NULL);
                it = changes.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Stop the callback if no more changes remain
    return changes.empty() ? G_SOURCE_REMOVE : G_SOURCE_CONTINUE;
}

// Bus callback to listen for EOS or ERROR
static gboolean bus_callback(GstBus *bus, GstMessage *msg, gpointer loop) {
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_main_loop_quit((GMainLoop*)loop);
            break;
        case GST_MESSAGE_ERROR: {
            GError *err;
            gchar *dbg;
            gst_message_parse_error(msg, &err, &dbg);
            g_printerr("Error: %s\n", err->message);
            g_error_free(err);
            g_free(dbg);
            g_main_loop_quit((GMainLoop*)loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    if (argc < 6 || argc > 7) {
        g_printerr("Usage: %s [<pipeline>] <source> <deformation file> <output> <times> <alphas>\n", argv[0]);
        return -1;
    }

    const gchar *pipeline_desc = argc == 7 ? argv[1] : NULL;
    const gchar *source = argv[argc == 7 ? 2 : 1];
    const gchar *deform_file = argv[argc == 7 ? 3 : 2];
    const gchar *output = argv[argc == 7 ? 4 : 3];
    gchar **times_str = g_strsplit(argv[argc == 7 ? 5 : 4], ",", -1);
    gchar **alphas_str = g_strsplit(argv[argc == 7 ? 6 : 5], ",", -1);

    gint n_times = g_strv_length(times_str);
    gint n_alphas = g_strv_length(alphas_str);

    if (n_times != n_alphas) {
        g_printerr("Times and alphas arrays must have the same length\n");
        g_strfreev(times_str);
        g_strfreev(alphas_str);
        return -1;
    }

    GstElement *src = nullptr, *demuxer = nullptr, *decoder = nullptr, *convert = nullptr, *sink = nullptr;

    if (pipeline_desc) {
        GError *error = NULL;
        pipeline = gst_parse_launch(pipeline_desc, &error);
        if (error) {
            g_printerr("Error parsing pipeline: %s\n", error->message);
            g_error_free(error);
            g_strfreev(times_str);
            g_strfreev(alphas_str);
            return -1;
        }

        // In this case, mozza should be part of pipeline_desc
        // Use gst_bin_get_by_name(GST_BIN(pipeline), "mozza") if mozza has a unique name
        mozza = gst_bin_get_by_name(GST_BIN(pipeline), "mozza");

    } else {
        pipeline = gst_pipeline_new("mozza-pipeline");
        src = gst_element_factory_make("filesrc", "source");
        demuxer = gst_element_factory_make("qtdemux", "demuxer");
        decoder = gst_element_factory_make("avdec_h264", "decoder");
        convert = gst_element_factory_make("videoconvert", "convert");
        mozza = gst_element_factory_make("mozza", "mozza");
        sink = gst_element_factory_make("filesink", "sink");

        if (!pipeline || !src || !demuxer || !decoder || !convert || !mozza || !sink) {
            g_printerr("Not all elements could be created\n");
            g_strfreev(times_str);
            g_strfreev(alphas_str);
            return -1;
        }

        g_object_set(src, "location", source, NULL);
        g_object_set(sink, "location", output, NULL);
        g_object_set(mozza, "deform-file", deform_file, NULL);

        gst_bin_add_many(GST_BIN(pipeline), src, demuxer, decoder, convert, mozza, sink, NULL);
        if (!gst_element_link(src, demuxer) ||
            !gst_element_link(decoder, convert) ||
            !gst_element_link(convert, mozza) ||
            !gst_element_link(mozza, sink)) {
            g_printerr("Elements could not be linked\n");
            gst_object_unref(pipeline);
            g_strfreev(times_str);
            g_strfreev(alphas_str);
            return -1;
        }

        g_signal_connect(demuxer, "pad-added", G_CALLBACK(on_pad_added), decoder);
    }

    // Store alpha changes
    for (gint i = 0; i < n_times; i++) {
        GstClockTime change_time = g_ascii_strtoull(times_str[i], NULL, 10) * GST_SECOND;
        gdouble alpha_value = g_ascii_strtod(alphas_str[i], NULL);

        AlphaChange ac { change_time, alpha_value };
        changes.push_back(ac);
    }

    g_strfreev(times_str);
    g_strfreev(alphas_str);

    // Set the pipeline to playing and run a main loop
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_print("Running...\n");

    GMainLoop *loop = g_main_loop_new(NULL, FALSE);

    // Add a bus watch to listen for EOS or ERROR
    GstBus *bus = gst_element_get_bus(pipeline);
    gst_bus_add_watch(bus, bus_callback, loop);
    gst_object_unref(bus);

    // Setup the timeout to update alpha values periodically (every 100 ms)
    if (!changes.empty()) {
        g_timeout_add(100, update_alpha, NULL);
    }

    g_main_loop_run(loop);

    // Cleanup
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    if (mozza) gst_object_unref(mozza);
    g_main_loop_unref(loop);

    return 0;
}