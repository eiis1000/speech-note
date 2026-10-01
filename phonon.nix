{ lib, python3, fetchurl, autoPatchelfHook, stdenv }:

# Keep Phonon's newer HF stack out of the main ASR interpreter. Wheels are pinned
# for this flake's x86_64-linux platform; no pip install happens at runtime.
let
  ps = python3.pkgs;
  wheel = { pname, version, url, sha256, dependencies ? [], native ? false }:
    ps.buildPythonPackage {
      inherit pname version dependencies;
      format = "wheel";
      src = fetchurl { inherit url sha256; };
      nativeBuildInputs = lib.optional native autoPatchelfHook;
      buildInputs = lib.optional native stdenv.cc.cc.lib;
      doCheck = false;
      dontUseNinjaBuild = true;
      dontUseNinjaInstall = true;
      # Fermion verifies its native libraries against bundled SHA-256 pins.
      dontStrip = true;
    };
  tokenizers = wheel {
    pname = "tokenizers";
    version = "0.23.1";
    url = "https://files.pythonhosted.org/packages/0d/d5/1353e5f677ec27c2494fb6a6725e82d56c985f53e90ec511369e7e4f02c6/tokenizers-0.23.1-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
    sha256 = "5075b405006415ea148a992d093699c66eb01952bf59f4d5727089a98bda45a4";
    dependencies = [ ps.huggingface-hub ];
    native = true;
  };
  safetensors = wheel {
    pname = "safetensors";
    version = "0.8.0";
    url = "https://files.pythonhosted.org/packages/28/50/f203ff3a3ddfe19308efc83c5a3a29ed02bf786732ec35e68bf9162f3365/safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
    sha256 = "fd6f3f93c9a0a7cc2788ee63fb763353d4bd2e89b0751bc78fcf7dda00bea774";
    native = true;
  };
  transformers = wheel {
    pname = "transformers";
    version = "5.17.0";
    url = "https://files.pythonhosted.org/packages/e8/d0/c502b60d684adbd98a8dc7d5bb866842772b816ac4354e4608be240041ae/transformers-5.17.0-py3-none-any.whl";
    sha256 = "78ec1ce21579b38dfb83950a0658cd119f87212a2fcfdff478096ce9d6c03801";
    dependencies = with ps; [ huggingface-hub numpy packaging pyyaml regex tqdm typer ]
      ++ [ tokenizers safetensors ];
  };
  fermion = wheel {
    pname = "fermion-research";
    version = "0.2.4";
    url = "https://files.pythonhosted.org/packages/ac/a1/c0ca0f85c6fc85ee296381aac292705382db9dfaa097e5cd1ebb4784f57d/fermion_research-0.2.4-py3-none-any.whl";
    sha256 = "a5442a940d83cd37071247ae87d418f52cd92e72e065ca68f0acdb3db3a03c68";
    dependencies = [ ps.torchWithRocm transformers ps.numpy ps.huggingface-hub ];
  };
in
python3.withPackages (_: [ fermion ps.soundfile ps.scipy ps.zstandard ])
