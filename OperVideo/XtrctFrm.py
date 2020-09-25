#
import cv2
import os
import time
import ffmpeg
import numpy
import cv2
import sys
import random


def XtrctFrm(video_path, trgt_path):
    #
    times = 0
    #
    frm_freq = 25
    #
    camera = cv2.VideoCapture(video_path)
    #
    while True:
        times += 1
        res, image = camera.read()
        if not res:
            print('not res , not image')
            break
        # if times % frm_freq == 0:
        #     #
        #     cv2.imwrite(trgt_path + str(times) + '.jpg', image)
        #     print(trgt_path + str(times) + '.jpg')
    print('图片提取结束:', times)
    camera.release()
    #
    return


def XtrctFrm2(video_path, trgt_path):
    #
    camera = cv2.VideoCapture(video_path)
    #
    for tmidx in range(1, 160000, 1000):
        #
        camera.set(0, tmidx)
        res, image = camera.read()
        # cv2.imwrite(trgt_path + str(tmidx) + '.jpg', image)
        print(trgt_path + str(tmidx) + '.jpg')
    #
    # while True:
    #     times += 1
    #     res, image = camera.read()
    #     if not res:
    #         print('not res , not image')
    #         break
    #     if times % frm_freq == 0:
    #         #
    #         cv2.imwrite(trgt_path + str(times) + '.jpg', image)
    #         print(trgt_path + str(times) + '.jpg')
    print('图片提取结束')
    camera.release()
    #
    return


def read_frame_as_jpeg(in_file, frame_num):
    out, err = (
        ffmpeg.input(in_file)
            .filter('select', 'gte(n,{})'.format(frame_num))
            .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
            .run(capture_stdout=True)
    )
    return out


def get_video_info(in_file):
    try:
        probe = ffmpeg.probe(in_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            print('No video stream found', file=sys.stderr)
            sys.exit(1)
        return video_stream
    except ffmpeg.Error as err:
        print(str(err.stderr, encoding='utf8'))
        sys.exit(1)


# if __name__ == '__main__':
#     file_path = '/Users/admin/Downloads/拜无忧.mp4'
#     video_info = get_video_info(file_path)
#     total_frames = int(video_info['nb_frames'])
#     print('总帧数：' + str(total_frames))
#     random_frame = random.randint(1, total_frames)
#     print('随机帧：' + str(random_frame))
#     out = read_frame_as_jpeg(file_path, random_frame)
#     image_array = numpy.asarray(bytearray(out), dtype="uint8")
#     image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#     cv2.imshow('frame', image)
#     cv2.waitKey()

if __name__ == '__main__':
    #
    start = time.clock()
    #
    # XtrctFrm2("F:\\Remote\\sample.flv", "F:\\Remote\\tmp\\")
    XtrctFrm2("F:\\Remote\\sample.mp4", "F:\\Remote\\tmp\\")
    #
    # out = read_frame_as_jpeg("F:\\Remote\\sample.flv", 1000)
    # image_array = numpy.asarray(bytearray(out), dtype="uint8")
    # image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # cv2.imshow('frame', image)
    # cv2.waitKey()
    #
    elapsed = time.clock() - start
    #
    print("Time used:", elapsed)
