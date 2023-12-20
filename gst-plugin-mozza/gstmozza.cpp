/* GStreamer
 * Copyright (C) 2021 Lara Kermarec <lara.git@kermarec.bzh> for CNRS
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Suite 500,
 * Boston, MA 02110-1335, USA.
 */
/**
 * SECTION:element-gstmozza
 *
 * The mozza element can modify the smile on a person's face.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 -v fakesrc ! mozza ! FIXME ! fakesink
 * ]|
 * FIXME Describe what the pipeline does.
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <dlib/opencv.h>

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>

#include <opencv2/opencv.hpp>

#include "gstmozza.h"

GST_DEBUG_CATEGORY_STATIC (gst_mozza_debug_category);
#define GST_CAT_DEFAULT gst_mozza_debug_category

/* prototypes */

static void gst_mozza_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_mozza_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);
static void gst_mozza_finalize (GObject * object);

static GstFlowReturn gst_mozza_transform_frame_ip (GstOpencvVideoFilter * filter,
    GstBuffer * buf, cv::Mat cv_img);

static void update_tracker_state(GstMozza* mozza, TrackerState state);

enum
{
  PROP_0,
  PROP_SHAPE_MODEL,
  PROP_DROP,
  PROP_USER_ID,
  PROP_DEFORM,
  PROP_ALPHA,
  PROP_FACE_THRESH,
  PROP_OVERLAY,
  PROP_OE_BETA,
  PROP_OE_FC,
};

/* pad templates */

#define VIDEO_SRC_CAPS \
    GST_VIDEO_CAPS_MAKE("{ RGB }")

#define VIDEO_SINK_CAPS \
    GST_VIDEO_CAPS_MAKE("{ RGB }")

#define DEFAULT_SHAPE_MODEL_PATH "/usr/share/dlib/shape_predictor_68_face_landmarks.dat"
#define DEFAULT_BETA 0.1
#define DEFAULT_FC 5.0

static const char* tracker_state_desc[] = {
  "TRACKER_OK: Good tracking",
  "TRACKER_UNINIT: Tracker hasn't been initialized",
  "TRACKER_NO_DEFORM: No deformation vector loaded",
  "TRACKER_NO_FACES: Failed to detect any faces",
  "TRACKER_MANY_FACES: Too many faces in frame",
  "TRACKER_START: [internal] invalid state",
};


/* class initialization */

G_DEFINE_TYPE_WITH_CODE (GstMozza, gst_mozza, GST_TYPE_OPENCV_VIDEO_FILTER,
  GST_DEBUG_CATEGORY_INIT (gst_mozza_debug_category, "mozza", 0,
  "debug category for mozza element"));

static void
gst_mozza_class_init (GstMozzaClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstOpencvVideoFilterClass *opencv_filter_class = GST_OPENCV_VIDEO_FILTER_CLASS (klass);

  init_mozza(gst_mozza_debug_category);

  gst_element_class_add_pad_template (GST_ELEMENT_CLASS(klass),
      gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
        gst_caps_from_string (VIDEO_SRC_CAPS)));
  gst_element_class_add_pad_template (GST_ELEMENT_CLASS(klass),
      gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
        gst_caps_from_string (VIDEO_SINK_CAPS)));

  gobject_class->set_property = gst_mozza_set_property;
  gobject_class->get_property = gst_mozza_get_property;
  gobject_class->finalize = gst_mozza_finalize;

  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS(klass),
      "Mozza", "Filter/Video", "Placeholder filter that adds facial mozza markers on the image",
      "Lara Kermarec <lara.git@kermarec.bzh>");

  g_object_class_install_property (gobject_class, PROP_ALPHA,
      g_param_spec_float ("alpha", "Smile intensity multiplicator",
          "Scales the intensity of the smile applied. Can be negative to apply a frown.",
          -10.f, 10.f, 1.f, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_SHAPE_MODEL,
      g_param_spec_string ("shape-model", "Landmark shape model",
          "Location of the shape model. You can get one from "
          "http://dlib.net/files/shape_predictor_68_face_mozza.dat.bz2",
          DEFAULT_SHAPE_MODEL_PATH, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_DROP,
      g_param_spec_boolean ("drop", "Drop frame on failure",
          "Sets whether to drop the frame when the filter fails to acquire a face.",
          FALSE, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_USER_ID,
      g_param_spec_string ("user-id", "User ID",
          "A unique identifier, used for debugging purposes.",
          "0", (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_DEFORM,
      g_param_spec_string ("deform", "Deformation vectors file",
          "Location of the deformation vectors file.",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_FACE_THRESH,
      g_param_spec_double ("face-thresh", "Face detection confidence threshold",
          "Confidence score threshold under which patterns detected by dlib's "
          "frontal_face_detector will be discarded. Adjust this parameter if "
          "you detect phantom faces or fail to detect valid faces.",
          0.0, 1.0, 0.15, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_OVERLAY,
      g_param_spec_boolean ("overlay", "Debug overlay toggle",
          "Enable the debug overlay, displaying facial landmark points and their deformations",
          FALSE, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_OE_BETA,
      g_param_spec_float ("beta", "One euro jitter filter beta",
          "If there is an issue with lagging landmarks during high speed "
          "movements, increase beta. For more detailed informations on tuning "
          "the One Euro Filter, visit http://cristal.univ-lille.fr/~casiez/1euro/",
          0.0, 1.0, DEFAULT_BETA, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_OE_FC,
      g_param_spec_float ("fc", "One euro jitter filter frequency cutoff",
          "If there is an issue with jittering landmarks at low speed, decrease fc. "
          "For more detailed informations on tuning the One Euro Filter, "
          "visit http://cristal.univ-lille.fr/~casiez/1euro/",
          0.0, 1000.0, DEFAULT_BETA, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  opencv_filter_class->cv_trans_ip_func =
      GST_DEBUG_FUNCPTR(gst_mozza_transform_frame_ip);

}

static void
gst_mozza_init (GstMozza *mozza)
{
  mozza->shape_model = g_strdup(DEFAULT_SHAPE_MODEL_PATH);
  mozza->shape_predictor = new dlib::shape_predictor;
  try {
    dlib::deserialize (mozza->shape_model) >> *mozza->shape_predictor;
  } catch (dlib::serialization_error &e) {
    GST_WARNING("Dlib shape model file not found at default location: %s",
        e.info.c_str());
    delete mozza->shape_predictor;
    mozza->shape_predictor = NULL;
  }
  mozza->face_detector =
      new dlib::frontal_face_detector(dlib::get_frontal_face_detector());
  mozza->drop = FALSE;
  mozza->alpha = 1.f;
  mozza->face_thresh = 0.15;
  mozza->overlay = FALSE;
  mozza->user_id = g_strdup("0");

  mozza->resize_rate = 1.f;

  mozza->mls = new ImgWarp_MLS_Rigid;
  mozza->mls->gridSize = MLS_GRID_SIZE;
  mozza->mls->preScale = true;
  mozza->mls->alpha = 1.4;  /* Important, no sane default for this one */

  mozza->oe_beta = DEFAULT_BETA;
  mozza->oe_fc = DEFAULT_FC;
  mozza->oe_filter = new OneEuroFilter(mozza->oe_beta, mozza->oe_fc);

  mozza->frame_count = 0;
  mozza->tracker_state = TRACKER_START;
  mozza->prev_time = 0;

  gst_opencv_video_filter_set_in_place (GST_OPENCV_VIDEO_FILTER_CAST (mozza),
      TRUE);
}

/* Actual frame processing done here */
static GstFlowReturn
gst_mozza_transform_frame_ip (GstOpencvVideoFilter * filter, GstBuffer * buf, cv::Mat cv_img)
{
  GstMozza *mozza = GST_MOZZA (filter);
  cv::Mat blurred_img;
  GstElement *element = GST_ELEMENT_CAST(filter);

  GST_DEBUG_OBJECT (mozza, "transform_frame_ip");

  GST_OBJECT_LOCK(mozza);
  mozza->frame_count++;

  if (!mozza->face_detector || !mozza->shape_predictor) {
    update_tracker_state(mozza, TRACKER_UNINIT);
    GST_OBJECT_UNLOCK(mozza);
    return GST_FLOW_OK;
  } else if (mozza->deformations.empty()) {
    update_tracker_state(mozza, TRACKER_NO_DEFORM);
    GST_OBJECT_UNLOCK(mozza);
    return GST_FLOW_OK;
  }

  /* Blur to try and counteract the CCD noise */
  const cv::Size BLUR_SIZE = cv::Size(5, 5);
  cv::GaussianBlur(cv_img, blurred_img, BLUR_SIZE, 0);

  std::vector<dlib::rectangle> dets;
  if (mozza->probable_face_ROI.empty())
    mozza->probable_face_ROI = cv::Rect(0, 0, cv_img.cols, cv_img.rows);
  find_faces(blurred_img, *mozza->face_detector, 0.15, dets, mozza->probable_face_ROI);

  /* If we fail to find any faces, simply drop the frame.
   * Crude but effective on the short short term.
   * Maybe Restart the stream after a second even if we can't reacquire? */
  if (dets.empty()) {
    GstFlowReturn res = mozza->drop ? GST_BASE_TRANSFORM_FLOW_DROPPED : GST_FLOW_OK;
    update_tracker_state(mozza, TRACKER_NO_FACES);
    return res;
    GST_OBJECT_UNLOCK(mozza);
  } else if (dets.size() > 1) {
    GstFlowReturn res = mozza->drop ? GST_BASE_TRANSFORM_FLOW_DROPPED : GST_FLOW_OK;
    update_tracker_state(mozza, TRACKER_MANY_FACES);
    GST_OBJECT_UNLOCK(mozza);
    return res;
  }

  std::vector<std::vector<cv::Point2f>> faces_pts;
  find_landmarks(blurred_img, *mozza->shape_predictor, dets, faces_pts);

  std::vector<cv::Point2f> landmarks;
  GstClock *clock = gst_element_get_clock(element);

  /* If gstreamer doesn't provide us with a clock, we simply skip the
   * stabilization step */
  if (!clock) {
    landmarks = faces_pts[0];
  } else {
    GstClockTime time = gst_clock_get_time(element->clock);
    GstClockTimeDiff frame_time = time - mozza->prev_time;
    mozza->oe_filter->update(frame_time / 1000000000.0, faces_pts[0], landmarks);
    mozza->prev_time = time;
  }

  std::vector<std::vector<cv::Point2f>> srcGroups;
  std::vector<std::vector<cv::Point2f>> dstGroups;
  if (!faces_pts.empty())
    get_deformation_groups(mozza->deformations, faces_pts[0],
        mozza->alpha, srcGroups, dstGroups);

  for (size_t i = 0; i < srcGroups.size(); i++)
    compute_MLS_on_ROI(cv_img, *mozza->mls, srcGroups[i], dstGroups[i]);

  if (mozza->overlay) {
    draw_debug(cv_img, srcGroups, dstGroups);
  }

  update_tracker_state(mozza, TRACKER_OK);

  GST_OBJECT_UNLOCK(mozza);

  return GST_FLOW_OK;
}

void
gst_mozza_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  GstMozza *mozza = GST_MOZZA (object);

  GST_DEBUG_OBJECT (mozza, "set_property");

  GST_OBJECT_LOCK(mozza);
  switch (property_id) {
    case PROP_SHAPE_MODEL:
      g_free(mozza->shape_model);
      if (mozza->shape_predictor)
        delete mozza->shape_predictor;
      mozza->shape_model = g_value_dup_string(value);
      mozza->shape_predictor = new dlib::shape_predictor;
      try {
        dlib::deserialize (mozza->shape_model) >> *mozza->shape_predictor;
      } catch (dlib::serialization_error &e) {
        GST_ERROR ("Error when deserializing landmark predictor model: %s",
            e.info.c_str());
        delete mozza->shape_predictor;
        mozza->shape_predictor = NULL;
      }
      break;
    case PROP_DROP:
      mozza->drop = g_value_get_boolean(value);
      break;
    case PROP_USER_ID:
      g_free(mozza->user_id);
      mozza->user_id = g_value_dup_string(value);
      break;
    case PROP_DEFORM:
      g_free(mozza->deform_file);
      mozza->deform_file = g_value_dup_string(value);
      import_deformations(mozza->deform_file, mozza->deformations);
      break;
    case PROP_ALPHA:
      mozza->alpha = g_value_get_float(value);
      break;
    case PROP_FACE_THRESH:
      mozza->face_thresh = g_value_get_double(value);
      break;
    case PROP_OVERLAY:
      mozza->overlay = g_value_get_boolean(value);
      break;
    case PROP_OE_BETA:
      mozza->oe_beta = g_value_get_float(value);
      break;
    case PROP_OE_FC:
      mozza->oe_fc = g_value_get_float(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
  GST_OBJECT_UNLOCK(mozza);
}

void
gst_mozza_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  GstMozza *mozza = GST_MOZZA (object);

  GST_DEBUG_OBJECT (mozza, "get_property");

  GST_OBJECT_LOCK(mozza);
  switch (property_id) {
    case PROP_SHAPE_MODEL:
      g_value_set_string (value, mozza->shape_model);
      break;
    case PROP_DROP:
      g_value_set_boolean(value, mozza->drop);
      break;
    case PROP_USER_ID:
      g_value_set_string(value, mozza->user_id);
      break;
    case PROP_DEFORM:
      g_value_set_string(value, mozza->deform_file);
      break;
    case PROP_ALPHA:
      g_value_set_float(value, mozza->alpha);
      break;
    case PROP_FACE_THRESH:
      g_value_set_double(value, mozza->face_thresh);
      break;
    case PROP_OVERLAY:
      g_value_set_boolean(value, mozza->overlay);
      break;
    case PROP_OE_BETA:
      g_value_set_float(value, mozza->oe_beta);
      break;
    case PROP_OE_FC:
      g_value_set_float(value, mozza->oe_fc);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
  GST_OBJECT_UNLOCK(mozza);
}

void
gst_mozza_finalize (GObject * object)
{
  GstMozza *mozza = GST_MOZZA (object);

  GST_DEBUG_OBJECT (mozza, "finalize");

  /* clean up object here */
  if (mozza->shape_predictor)
    delete mozza->shape_predictor;
  if (mozza->face_detector)
    delete mozza->face_detector;
  if (mozza->mls)
    delete mozza->mls;
  if (mozza->oe_filter)
    delete mozza->oe_filter;

  g_free(mozza->shape_model);
  g_free(mozza->user_id);
  g_free(mozza->deform_file);

  mozza->deformations.~std::vector<Deformation>();

  G_OBJECT_CLASS (gst_mozza_parent_class)->finalize (object);
}

static void
update_tracker_state(GstMozza* mozza, TrackerState state)
{
  if (state != mozza->tracker_state) {
    mozza->tracker_state = state;
    GST_WARNING("frame: %lu, user-id: %s, %s",mozza->frame_count, mozza->user_id, tracker_state_desc[state]);
  }
}

static gboolean
plugin_init (GstPlugin * plugin)
{

  /* FIXME Remember to set the rank if it's an element that is meant
     to be autoplugged by decodebin. */
  return gst_element_register (plugin, "mozza", GST_RANK_NONE,
      GST_TYPE_MOZZA);
}

/* FIXME: these are normally defined by the GStreamer build system.
   If you are creating an element to be included in gst-plugins-*,
   remove these, as they're always defined.  Otherwise, edit as
   appropriate for your external plugin package. */
#ifndef VERSION
#define VERSION "0.0.FIXME"
#endif
#ifndef PACKAGE
#define PACKAGE "FIXME_package"
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "FIXME_package_name"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "http://FIXME.org/"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    mozza,
    "Placeholder filter that adds facial mozza markers on the image",
    plugin_init, VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)

