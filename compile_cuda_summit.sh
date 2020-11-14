#!/bin/bash


 module load gcc cuda cmake/3.15.2

 rm -r CMakeFiles
 rm cmake_install.cmake CPackConfig.cmake CPackSourceConfig.cmake CMakeCache.txt 
 cmake -DWITH_OMP=off
 make 

