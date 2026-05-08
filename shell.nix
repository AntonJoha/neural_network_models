{ pkgs ? import <nixpkgs> { } }:

let
  pythonEnv = pkgs.python3.withPackages (ps: with ps; [
    numpy
    torch
    ruff
    gymnasium
  ]);
in
pkgs.mkShell {
  packages = [
    pythonEnv
  ];
}
