project(
    'bearly',
    'cpp',
    default_options: [
        'warning_level=2',
        'cpp_std=c++20',
    ],
)

py = import('python').find_installation()
nanoarrow_dep = dependency(
    'nanoarrow',
    default_options: ['default_library=static'],
)
nanobind_dep = dependency('nanobind')

py.extension_module(
    'bearly',
    sources: ['src/bearly/bearly_ext.cc'],
    dependencies: [nanoarrow_dep, nanobind_dep],
)
