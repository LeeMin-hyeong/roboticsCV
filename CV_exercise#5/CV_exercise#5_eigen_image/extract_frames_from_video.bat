del keyframes\* /Q
ffmpeg -i light.mpg -y -an -sameq -f image2 -r 30 -ss 00:00:00 -t 00:00:05 keyframes/frame_%%06d.jpg