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
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifndef _GST_MOZZA_H_
#define _GST_MOZZA_H_

#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include <gst/opencv/gstopencvvideofilter.h>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

#include "mozza.h"

G_BEGIN_DECLS

#define GST_TYPE_MOZZA   (gst_mozza_get_type())
#define GST_MOZZA(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_MOZZA,GstMozza))
#define GST_MOZZA_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_MOZZA,GstMozzaClass))
#define GST_IS_MOZZA(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_MOZZA))
#define GST_IS_MOZZA_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_MOZZA))

typedef struct _GstMozza GstMozza;
typedef struct _GstMozzaClass GstMozzaClass;

enum TrackerState {
  TRACKER_OK,
  TRACKER_UNINIT,
  TRACKER_NO_DEFORM,
  TRACKER_NO_FACES,
  TRACKER_MANY_FACES,
  TRACKER_START,
};

struct _GstMozza
{
  GstOpencvVideoFilter base_mozza;

  gchar *shape_model;
  gchar *user_id;
  gchar *deform_file;
  gboolean drop;
  gfloat alpha;
  gdouble face_thresh;
  gboolean overlay;
  gfloat oe_beta;
  gfloat oe_fc;

  dlib::frontal_face_detector *face_detector;
  dlib::shape_predictor *shape_predictor;

  std::vector<Deformation> deformations;
  float resize_rate;
  ImgWarp_MLS_Rigid *mls;
  OneEuroFilter *oe_filter;
  uint64_t frame_count;
  TrackerState tracker_state;
  GstClockTime prev_time;
  cv::Rect probable_face_ROI;
};

struct _GstMozzaClass
{
  GstOpencvVideoFilterClass base_mozza_class;
};

GType gst_mozza_get_type (void);

G_END_DECLS

#endif
