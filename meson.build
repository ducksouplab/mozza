project('mozza', 'c', 'cpp',
  version : '1.18.0.1',
  default_options : ['warning_level=3', 'cpp_std=c++14'],
  meson_version : '>= 0.40.1')

cxx = meson.get_compiler('cpp')

sse2_args = '-msse2'
sse41_args = '-msse4.1'
avx_args = '-mavx'

cxx_args = []
if host_machine.cpu_family().startswith('x86')
  if cxx.has_argument(sse2_args)
    cxx_args += sse2_args
  endif
  if cxx.has_argument(sse41_args)
    cxx_args += sse41_args
  endif
  if cxx.has_argument(avx_args)
    cxx_args += avx_args
  endif
endif

gst_version = meson.project_version()
version_arr = gst_version.split('.')
gst_version_major = version_arr[0]
gst_version_minor = version_arr[1]
gst_version_micro = version_arr[2]
if version_arr.length() == 4
  gst_version_nano = version_arr[3]
else
  gst_version_nano = 0
endif

gst_req = '>= @0@.@1@.0'.format(gst_version_major, gst_version_minor)

plugins_install_dir = '@0@/gstreamer-1.0'.format(get_option('libdir'))

opencv_dep = dependency('opencv', version : '>= 3.0.0', required: false)
if not opencv_dep.found()
  opencv_dep = dependency('opencv4', version : '>= 4.0.0', required: true)
endif
dlib_dep = dependency('dlib-1', required : true)
gstbase_dep = dependency('gstreamer-base-1.0', version : gst_req,
  fallback : ['gstreamer', 'gst_base_dep'])
gstvideo_dep = dependency('gstreamer-video-1.0', version : gst_req,
  fallback : ['gst-plugins-base', 'video_dep'])
gstopencv_dep = cxx.find_library('gstopencv-1.0', required : true)


subdir('lib')
subdir('templater')
subdir('gst-plugin-mozza')

executable('mozza-templater',
  [templater_src, lib_src],
  dependencies : [opencv_dep, dlib_dep, imgwarp_dep],
  include_directories: inc,
  cpp_args : [cxx_args],
  install : true)

library('gstmozza',
  [gst_mozza_src, lib_src],
  dependencies : [gstbase_dep, gstvideo_dep, gstopencv_dep, opencv_dep, dlib_dep, imgwarp_dep],
  include_directories: inc,
  cpp_args : [cxx_args, mozza_args],
  install : true,
  install_dir : plugins_install_dir)
