#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to interface opencv functions

Hengen Lab
Washington University in St. Louis
Author: Kiran Bhaskaran-Nair
Email: kbn.git@gmail.com
Version:  0.1


List of functions/class in ntk_videos
class NTKVideos
make_video_from_images(imgpath, videopath, videofilename,
                       imgext='.jpg', codec='XVID', v_fps=30)
natural_sort(list_unsorted):

'''


try:
    import cv2
except ImportError:
    raise ImportError('conda install ' +
                      '-c https://conda.binstar.org/menpo opencv')
import os


class NTKVideos:

    '''
    ntk video class, interface to opencv

    get video attributes
    Example:
    videofilename = '/home/user/e3v810a-20190307T0740-0840.mp4'
    lstream = 0
    lstream is 1 is video is streaming or 0 if video is already saved
    v  = NTKVideos(videofilename, lstream)
    v contains length, width, height information from video

    play video, please press q to exit
    v.play_video()

    extract all frames
    outpath = '/home/user/out/'
    v.extract_frames(outpath)

    grab a frame
    grab_frame_num(self, frame_num, pathout)
    frame_num = 1
    outpath = '/home/user/out/'
    # To write frame to disk
    v.grab_frame_num(frame_num, outpath)
    # To see the frame, keep outpath empty
    v.grab_frame_num(frame_num)
    '''

    def __init__(self, name, lstream):
        self.cap = cv2.VideoCapture(name)

        # Check if camera opened successfully
        if self.cap.isOpened() is True:
            if lstream == 0:
                self.length = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            codec = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            self.codec = [chr((codec >> 8 * i) & 0xFF) for i in range(4)]
            self.brightness = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
            self.contrast = self.cap.get(cv2.CAP_PROP_CONTRAST)

        else:
            print("Error opening video stream or file")

    def play_video(self):

        '''
        videofilename = '/home/user/e3v810a-20190307T0740-0840.mp4'
        lstream = 0
        lstream is 1 is video is streaming or 0 if video is already saved
        v  = NTKVideos(videofilename, lstream)
        v contains length, width, height information from video

        play video, please press q to exit
        v.play_video()
        '''

        print("Please press q to exit")
        while True:
            # Capture frame-by-frame
            self.ret, self.frame = self.cap.read()

            if self.ret is True:
                cv2.imshow('video', self.frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        # self.cap.release()
        cv2.destroyAllWindows()

    def extract_frames(self, pathout):

        '''
        videofilename = '/home/user/e3v810a-20190307T0740-0840.mp4'
        lstream = 0
        lstream is 1 is video is streaming or 0 if video is already saved
        v  = NTKVideos(videofilename, lstream)
        v contains length, width, height information from video

        extract all frames and write to disk
        outpath = '/home/user/out/'
        v.extract_frames(outpath)
        '''

        frame_num = 0
        while True:
            frame_num = frame_num + 1

            # Capture frame-by-frame
            self.ret, self.frame = self.cap.read()

            if self.ret is True:
                cv2.imwrite(os.path.join(pathout, "frame%d.jpg") %
                            frame_num, self.frame)
            else:
                break

        # self.cap.release()
        cv2.destroyAllWindows()

    def grab_frame_num(self, frame_num, pathout=None):

        '''
        videofilename = '/home/user/e3v810a-20190307T0740-0840.mp4'
        lstream = 0
        lstream is 1 is video is streaming or 0 if video is already saved
        v  = NTKVideos(videofilename, lstream)
        v contains length, width, height information from video

        grab a frame
        grab_frame_num(self, frame_num, pathout)
        frame_num = 1
        outpath = '/home/user/out/'

        # To write frame to disk
        v.grab_frame_num(frame_num, outpath)

        # To see the frame, keep outpath empty
        v.grab_frame_num(frame_num)
        '''

        self.cap.set(1, frame_num)
        self.ret, self.frame = self.cap.read()

        if self.ret is True:
            if pathout:
                cv2.imwrite(os.path.join(pathout, "frame%d.jpg") %
                            frame_num, self.frame)
            else:
                cv2.imshow('video', self.frame)
        else:
            print('Error could not get frame')

        # self.cap.release()
        # cv2.destroyAllWindows()


def make_video_from_images(imgpath, videopath, videofilename,
                           imgext='.jpg', codec='XVID', v_fps=30):

    '''
    imgpath where images are saved
    videopath where output video to be saved
    videofilename filename of the output video
    imgext default '.jpg'
    codec default 'XVID'
    v_fps default 30

    make_video_from_images(imgpath, videopath, videofilename,
        imgext='.jpg', codec='XVID',v_fps=30)
    ntk.make_video_from_images('/home/user/img/', '/home/user/img/out/',
        'video2.avi', imgext='.jpg', codec='XVID', v_fps=30)
    '''

    img = []
    imgcounter = 0
    files = natural_sort(os.listdir(imgpath))
    for f in files:
        if f.endswith(imgext):
            # print(f)
            # print(imgcounter)
            img.append(cv2.imread(str(imgpath) + f))
            if img[imgcounter] is None:
                print('Skipping frame', imgcounter)
                imgcounter = imgcounter + 1
            else:
                # print(img)
                if imgcounter == 0:
                    # Save first frames height and width
                    height, width, layers = img[imgcounter].shape
                    h0 = height
                    w0 = width
                    fourcc = cv2.VideoWriter_fourcc(*str(codec))
                    video = cv2.VideoWriter(
                              (str(videopath) + str(videofilename)),
                              fourcc, float(v_fps), (width, height))
                # print(imgcounter)

                height, width, layers = img[imgcounter].shape
                if h0 != height and w0 != width:
                    print('Image ', imgcounter,
                          'resized, please check the video')
                    img[imgcounter] = cv2.resize(img[imgcounter], (h0, w0))

                video.write(img[imgcounter])
                imgcounter = imgcounter + 1

    cv2.destroyAllWindows()
    video.release()


def natural_sort(list_unsorted):
    import re

    '''
    returns naturally sorted list
    '''

    return sorted(list_unsorted,
                  key=lambda nats: [int(s) if s.isdigit() else s.lower()
                                    for s in re.split(r'(\d+)', nats)])


if __name__ == '__main__':
    videofilename = 'e3v810a-20190307T0740-0840.mp4'
    # videofilename = 0
    lstream = 0
    v = NTKVideos(videofilename, lstream)
