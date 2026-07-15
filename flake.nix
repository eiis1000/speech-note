{
  description = "Local speech-to-text note taker";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  inputs.crispasr-src = {
    url = "github:CrispStrobe/CrispASR/f23d948562a77e3581c959a034d14a1ac8fb89fd";
    flake = false;
  };

  outputs =
    {
      self,
      nixpkgs,
      crispasr-src,
      ...
    }:
    let
      system = "x86_64-linux";
      overlay = final: _prev: {
        inherit (final.callPackages ./packages.nix { inherit crispasr-src; })
          crispasr-vulkan
          speech-note
          speech-note-python
          ;
      };
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        overlays = [ overlay ];
      };
    in
    {
      overlays.default = overlay;

      packages.${system} = {
        default = pkgs.speech-note;
        speech-note = pkgs.speech-note;
        speech-note-python = pkgs.speech-note-python;
        llama-cpp-vulkan = pkgs.llama-cpp-vulkan;
        crispasr-vulkan = pkgs.crispasr-vulkan;
        whisper-cpp-vulkan = pkgs.whisper-cpp-vulkan;
      };

      apps.${system}.default = {
        type = "app";
        program = "${pkgs.speech-note}/bin/speech-note";
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          speech-note
          speech-note-python
          ffmpeg
          portaudio
          crispasr-vulkan
          llama-cpp-vulkan
          vulkan-tools
          whisper-cpp-vulkan
          wl-clipboard
        ];

        shellHook = ''
          unset PYTHONPATH PYTHONHOME VIRTUAL_ENV __PYVENV_LAUNCHER__
          export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="''${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
          echo "speech-note dev shell"
          echo "Installed tool:   speech-note --help   (runs the Nix-store copy)"
          echo "Working tree run: python -m speech_note --help"
          echo "Tests:            python -m unittest discover -s tests"
        '';
      };
    };
}
