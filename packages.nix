{
  lib,
  stdenv,
  cmake,
  pkg-config,
  shaderc,
  vulkan-headers,
  vulkan-loader,
  spirv-headers,
  python3,
  ffmpeg,
  portaudio,
  llama-cpp-vulkan,
  vulkan-tools,
  whisper-cpp-vulkan,
  wl-clipboard,
  writeShellApplication,
  crispasr-src,
}:

let
  pythonEnv = python3.withPackages (
    ps: with ps; [
      faster-whisper
      huggingface-hub
      librosa
      numpy
      onnx-asr
      pip
      pocketsphinx
      requests
      sherpa-onnx
      sounddevice
      setuptools
      torchWithRocm
      transformers
      webrtcvad
      wheel
    ]
  );

  crispAsrVulkan = stdenv.mkDerivation {
    pname = "crispasr-vulkan";
    version = "0.6.9-f23d9485";
    src = crispasr-src;

    nativeBuildInputs = [
      cmake
      pkg-config
      shaderc
    ];
    buildInputs = [
      vulkan-headers
      vulkan-loader
    ];

    cmakeFlags = [
      "-DCMAKE_BUILD_TYPE=Release"
      "-DBUILD_SHARED_LIBS=ON"
      "-DCRISPASR_BUILD_TESTS=OFF"
      "-DCRISPASR_BUILD_EXAMPLES=ON"
      "-DCRISPASR_BUILD_SERVER=OFF"
      "-DGGML_VULKAN=ON"
      "-DGGML_CCACHE=OFF"
      "-DVulkan_INCLUDE_DIR=${vulkan-headers}/include"
      "-DVulkan_LIBRARY=${vulkan-loader}/lib/libvulkan.so"
    ];
    NIX_CFLAGS_COMPILE = "-I${spirv-headers}/include";
  };

  runtimeSrc = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./speech_note
      ./tools
    ];
  };

  speechNote = writeShellApplication {
    name = "speech-note";
    runtimeInputs = [
      ffmpeg
      pythonEnv
      portaudio
      crispAsrVulkan
      llama-cpp-vulkan
      vulkan-tools
      whisper-cpp-vulkan
      wl-clipboard
    ];
    text = ''
      unset PYTHONHOME VIRTUAL_ENV __PYVENV_LAUNCHER__
      export PYTHONPATH=${runtimeSrc}
      export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="''${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
      exec ${pythonEnv}/bin/python -m speech_note "$@"
    '';
  };
in
{
  crispasr-vulkan = crispAsrVulkan;
  speech-note = speechNote;
  speech-note-python = pythonEnv;
}
