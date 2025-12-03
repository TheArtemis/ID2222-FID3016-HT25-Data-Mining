#!/bin/bash

#check for installation of parallel-rsync

if [ "$#" -ne 1 ] ; then
	echo "Data file not supplied."
	echo "Usage ./plot {data-file.txt}"
	exit
fi

gnuplot -e "filename='$1'" graph.gnuplot

# Open the generated image with a platform-appropriate command
if command -v xdg-open >/dev/null 2>&1; then
	xdg-open graph.png
elif command -v gnome-open >/dev/null 2>&1; then
	gnome-open graph.png
elif command -v open >/dev/null 2>&1; then
	open graph.png
else
	echo "Generated graph: graph.png"
	echo "No known image opener found (tried xdg-open, gnome-open, open)."
	echo "You can open it manually, e.g. 'open graph.png' on macOS or install xdg-utils on Linux."
	exit 1
fi
