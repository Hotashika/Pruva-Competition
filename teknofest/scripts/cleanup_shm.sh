#!/bin/bash

echo "[CLEANUP] Shared Memory cleaning..."
rm -f /dev/shm/RGB_DATA
rm -f /dev/shm/DEPTH_DATA
rm -f /dev/shm/ZED_META
rm -f /dev/shm/ZED_IMU
rm -f /dev/shm/ZED_CALIB
echo "[CLEANUP] Cleaning completed."
