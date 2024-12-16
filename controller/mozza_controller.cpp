#include <gst/gst.h>
#include <gst/controller/gstcontroller.h>
#include <gst/controller/gstinterpolationcontrolsource.h>
#include <iostream>

void on_pad_added(GstElement *element, GstPad *pad, gpointer data);

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

    GstElement *pipeline = nullptr;
    GstElement *src = nullptr, *demuxer = nullptr, *decoder = nullptr, *convert = nullptr, *mozza = nullptr, *sink = nullptr;

    if (pipeline_desc) {
        GError *error = NULL;
        pipeline = gst_parse_launch(pipeline_desc, &error);
        if (error) {
            g_printerr("Error parsing pipeline: %s\n", error->message);
            g_error_free(error);
            return -1;
        }
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
            return -1;
        }

        g_object_set(src, "location", source, NULL);
        g_object_set(sink, "location", output, NULL);
        g_object_set(mozza, "deform-file", deform_file, NULL);

        gst_bin_add_many(GST_BIN(pipeline), src, demuxer, decoder, convert, mozza, sink, NULL);
        if (!gst_element_link(src, demuxer) || !gst_element_link(decoder, convert) || !gst_element_link(convert, mozza) || !gst_element_link(mozza, sink)) {
            g_printerr("Elements could not be linked\n");
            gst_object_unref(pipeline);
            return -1;
        }

        g_signal_connect(demuxer, "pad-added", G_CALLBACK(on_pad_added), decoder);
    }

    GstControlSource *control_source = gst_interpolation_control_source_new();
    GstInterpolationControlSource *alpha_control_source = GST_INTERPOLATION_CONTROL_SOURCE(control_source);

    if (!alpha_control_source) {
        g_printerr("Failed to create control source\n");
        gst_object_unref(pipeline);
        g_strfreev(times_str);
        g_strfreev(alphas_str);
        return -1;
    }

    gst_object_add_control_binding(GST_OBJECT(mozza),
        gst_direct_control_binding_new(GST_OBJECT(mozza), "alpha", GST_CONTROL_SOURCE(alpha_control_source)));

    for (gint i = 0; i < n_times; i++) {
        GstClockTime time = g_ascii_strtoull(times_str[i], NULL, 10) * GST_SECOND;
        gdouble alpha = g_ascii_strtod(alphas_str[i], NULL);

        gst_timed_value_control_source_set(GST_TIMED_VALUE_CONTROL_SOURCE(alpha_control_source), time, alpha);
    }

    g_strfreev(times_str);
    g_strfreev(alphas_str);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_print("Running...\n");
    gst_element_get_state(pipeline, NULL, NULL, -1);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    return 0;
}

void on_pad_added(GstElement *element, GstPad *pad, gpointer data) {
    GstElement *decoder = (GstElement *)data;
    GstPad *sinkpad = gst_element_get_static_pad(decoder, "sink");
    gst_pad_link(pad, sinkpad);
    gst_object_unref(sinkpad);
}
