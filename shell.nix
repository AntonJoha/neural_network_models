{ }:

let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-26.05.tar.gz") {};
  
  python = pkgs.python3.withPackages (ps: with ps; [
    jupyterlab
    ipykernel
    matplotlib
    numpy
    pandas
    torch
    gymnasium
    pip
    ruff


  ]);

in
pkgs.mkShell {
  buildInputs = [
    python
    pkgs.geos
    pkgs.gdal
    pkgs.proj
  ];

  packages = [
    python
  ];
  shellHook = ''
    # Install the project in editable mode so imports resolve correctly.
    export PYTHONPATH=$PWD/src:$PYTHONPATH
    '';
}
