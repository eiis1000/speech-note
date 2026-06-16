{
  description = "Local speech-to-text note taker";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  inputs.crispasr-src = {
    url = "github:CrispStrobe/CrispASR/f23d948562a77e3581c959a034d14a1ac8fb89fd";
    flake = false;
  };

  outputs = { self, nixpkgs, crispasr-src, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      python = pkgs.python3;
      pythonEnv = python.withPackages (ps: with ps; [
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
      ]);
      llamaCpp = pkgs.llama-cpp-vulkan;
      whisperCpp = pkgs.whisper-cpp-vulkan;
      crispAsrVulkan = pkgs.stdenv.mkDerivation {
        pname = "crispasr-vulkan";
        version = "0.6.9-f23d9485";
        src = crispasr-src;

        nativeBuildInputs = [
          pkgs.cmake
          pkgs.pkg-config
          pkgs.shaderc
        ];
        buildInputs = [
          pkgs.vulkan-headers
          pkgs.vulkan-loader
        ];

        cmakeFlags = [
          "-DCMAKE_BUILD_TYPE=Release"
          "-DBUILD_SHARED_LIBS=ON"
          "-DCRISPASR_BUILD_TESTS=OFF"
          "-DCRISPASR_BUILD_EXAMPLES=ON"
          "-DCRISPASR_BUILD_SERVER=OFF"
          "-DGGML_VULKAN=ON"
          "-DGGML_CCACHE=OFF"
          "-DVulkan_INCLUDE_DIR=${pkgs.vulkan-headers}/include"
          "-DVulkan_LIBRARY=${pkgs.vulkan-loader}/lib/libvulkan.so"
        ];
        NIX_CFLAGS_COMPILE = "-I${pkgs.spirv-headers}/include";
      };
      runtimeSrc = pkgs.lib.fileset.toSource {
        root = ./.;
        fileset = pkgs.lib.fileset.unions [
          ./speech_note
          ./tools
        ];
      };
      runtimeInputs = [
        pkgs.ffmpeg
        pythonEnv
        pkgs.portaudio
        crispAsrVulkan
        llamaCpp
        pkgs.vulkan-tools
        whisperCpp
        pkgs.wl-clipboard
      ];
      speechNote = pkgs.writeShellApplication {
        name = "speech-note";
        inherit runtimeInputs;
        text = ''
          unset PYTHONHOME VIRTUAL_ENV __PYVENV_LAUNCHER__
          export PYTHONPATH=${runtimeSrc}
          export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="''${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
          exec ${pythonEnv}/bin/python -m speech_note "$@"
        '';
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = runtimeInputs ++ [ speechNote ];

        shellHook = ''
          unset PYTHONPATH PYTHONHOME VIRTUAL_ENV __PYVENV_LAUNCHER__
          export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="''${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
          echo "speech-note dev shell"
          echo "Installed tool:   speech-note --help   (runs the Nix-store copy)"
          echo "Working tree run: python -m speech_note --help"
          echo "Tests:            python -m unittest discover -s tests"
        '';
      };

      packages.${system} = {
        default = speechNote;
        speech-note = speechNote;
        llama-cpp-vulkan = llamaCpp;
        crispasr-vulkan = crispAsrVulkan;
        whisper-cpp-vulkan = whisperCpp;
      };

      apps.${system}.default = {
        type = "app";
        program = "${speechNote}/bin/speech-note";
      };
    };
}
