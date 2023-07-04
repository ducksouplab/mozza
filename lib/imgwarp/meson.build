imgwarp = library('imgwarp',
           'imgwarp_piecewiseaffine.cpp',
           'imgwarp_mls_rigid.cpp',
           'imgwarp_mls_similarity.cpp',
           'delaunay.cpp',
           'imgwarp_mls.cpp',
           install : true,
           dependencies : [opencv_dep])

imgwarp_dep = declare_dependency(link_with: imgwarp,
    include_directories : ['.'],
    dependencies : [opencv_dep])
