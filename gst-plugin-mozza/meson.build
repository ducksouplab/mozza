gst_mozza_src = files([
  'gstmozza.cpp',
])

mozza_args = ['-DUSE_GST_LOGGING']
if host_machine.cpu_family().startswith('x86')
  mozza_args += '-DUSE_SSE41 -DUSE_SSE4_INSTRUCTIONS=ON'
endif
