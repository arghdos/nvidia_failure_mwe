################################################################################
#                                                                              #
#                           nvidia_test.py                                     #
#                                                                              #
#                                                                              #
# A simple test to demonstrate the failure of NVIDIA's OpenCL runtime packaged #
# with the RHEL-6 Linux x65 GPU driver version 375.26 on a Tesla K40m GPU      #
#                                                                              #
# Requires the an install of said NVIDIA driver, and at least one other user   #
# specifed OpenCL driver.  Additionally compatible with ICD-loaders.           #
#                                                                              #
#                          Nicholas Curtis, 2018                               #
################################################################################

from argparse import ArgumentParser
import subprocess
from tempfile import TemporaryFile
import os


def compile_and_test(header_path, lib_path, platform_name, defines='',
                     should_fail=False, lib_name='OpenCL'):
    # define the compilation string
    compilation_template = (
        'gcc -fPIC -O0 -g -std=c99 -xc {defines} jacobian_kernel_main.ocl '
        'jacobian_kernel_compiler.ocl timer.ocl read_initial_conditions.ocl '
        'ocl_errorcheck.ocl -I{header_path} -Wl,-rpath,{lib_path} -l{lib_name} -o '
        'test.out')

    # normalize the libpath / test for presence of the OpenCL library:
    lib_path = os.path.realpath(lib_path)
    assert os.path.isdir(lib_path), (
        'Library directory {} for platform {} not found'.format(
            lib_path, platform_name))

    # normalize the header path / test for existance
    header_path = os.path.realpath(header_path)
    assert os.path.isdir(header_path), ('OpenCL headers not found at:\n {}'.format(
        header_path))

    # turn defines into preprocessor macros
    if defines:
        defines = ' '.join('-D{}'.format(x) for x in defines.split())

    # transform call
    compilation_template = compilation_template.format(
        defines=defines, header_path=header_path, lib_path=lib_path,
        lib_name=lib_name)

    def run(cmd):
        # now compile / run
        with TemporaryFile() as stdout:
            with TemporaryFile() as stderr:
                try:
                    print(' '.join(cmd))
                    subprocess.check_call(cmd, stdout=stdout, stderr=stderr)
                except subprocess.CalledProcessError as e:
                    print('Error compiling or running: {}'.format(' '.join(e.cmd)))
                    stderr.seek(0)
                    raise Exception(str(stderr.read()))
            stdout.seek(0)
            output = stdout.read()
        return str(output)

    # compile
    run(compilation_template.split())

    # look for the sum the forward stoichometic coefficients - 1 of reaction 1988
    # (0-based) in the output.
    output = run(['./test.out', platform_name, '896', '1', '1'])

    # The correct answer is '0' (4 * 0 + 1 - 1 = 0)
    # The incorrect answer is '-1'
    # if we expect this call to fail (i.e., NVIDIA w/o the PRINT macro defined)
    # we use -1, else 0
    desired = 'rxn:1988, spec:349, nu_fwd_sum:{val}, nu_rev_sum:3'
    val = -1 if should_fail else 0
    desired = desired.format(val=val)

    assert desired in output, 'Test of {} failed'.format(platform_name)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-nv', '--nvidia_path',
                        type=str,
                        required=True,
                        help='The path to the NVIDIA OpenCL runtime library ('
                             'libOpenCL.so). Typically this is stored in /usr/lib64/'
                             ' or /usr/lib/, and should also contain NVIDIA driver '
                             'libraries, e.g., libnvidia-compiler.so.375.26, and '
                             'libnvidia-opencl.so.384.81.')
    parser.add_argument('-hp', '--header_path',
                        type=str,
                        required=True,
                        help="The path to the OpenCL headers. These come bundled "
                             "with some OpenCL packages, e.g., NVIDIA's are "
                             "typically stored in /usr/local/cuda/include/ or the "
                             "like.  The source of the OpenCL headers usually "
                             "doesn't matter, but the official ones are available "
                             "from the [Khronos repo](https://github.com/KhronosGroup/OpenCL-Headers/tree/master/opencl12/CL) " # noqa
                             "if desired.")
    parser.add_argument('-op', '--other_opencl_libpath',
                        type=str,
                        required=True,
                        help="The path to another OpenCL runtime library's "
                             "libOpenCL.so.  Typical paths look like "
                             "/opt/intel/opencl/lib64/ for Intel, or "
                             "/usr/local/lib64/ for POCL, or "
                             "/opt/AMDAPPSDK-3.0/lib/x86_64/sdk/ for AMD.")
    parser.add_argument('-ol', '--other_opencl_libname',
                        type=str,
                        required=False,
                        default='OpenCL',
                        help='The name of the OpenCL library to link against for'
                             'the other platform')
    parser.add_argument('-on', '--other_opencl_platform_name',
                        type=str,
                        required=True,
                        help="The platform name (or substring thereof) reported by "
                             "the desired other OpenCL runtime.  Common choice "
                             "include 'Intel' for Intel, 'AMD' for AMD, or "
                             "'Portable Computing Language' for POCL.")

    args = parser.parse_args()

    # test nvidia w/o printing -- should fail
    compile_and_test(args.header_path, args.nvidia_path, 'NVIDIA',
                     should_fail=True)

    # finally, turn on printing for NVIDIA -- should pass
    compile_and_test(args.header_path, args.nvidia_path, 'NVIDIA', defines='PRINT',
                     should_fail=True)

    # test other OpenCL implementation -- should pass
    compile_and_test(args.header_path, args.other_opencl_libpath,
                     args.other_opencl_platform_name, defines='',
                     should_fail=False, lib_name=args.other_opencl_libname)
