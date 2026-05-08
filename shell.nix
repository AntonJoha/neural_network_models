{ pkgs ? import <nixpkgs> { } }:

let
  pythonEnv = pkgs.python3.withPackages (ps: with ps; [
    numpy
    torch
    gymnasium
  ]);
in
pkgs.mkShell {
  packages = [
    pythonEnv
  ];
}
