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
    raise ImportError('\tpip install opencv-python\n')
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
            self.name = name
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

    def rotate_video(self, angle, pathout):

        '''
        videofilename = '/home/user/e3v810a-20190307T0740-0840.mp4'
        lstream = 0
        lstream is 1 is video is streaming or 0 if video is already saved
        v  = NTKVideos(videofilename, lstream)
        v contains length, width, height information from video

        rotate video
        angle = angle to rotate the video
        outpath = '/home/user/out/', where to save new files
        v.rotate_video(angle, outpath)
        '''

        import os.path as op

        img = []
        img_c = 0
        videofilename = (op.splitext(self.name)[0] + str("_r") +
                         op.splitext(self.name)[1])
        print("angle ", angle)
        while True:
            # Capture frame-by-frame
            self.ret, self.frame = self.cap.read()

            if self.ret is True:
                M = cv2.getRotationMatrix2D((int(self.width/2),
                                            int(self.height/2)),
                                            int(angle), 1)
                # please check the flags
                img.append(cv2.warpAffine(self.frame, M,
                                          (int(self.width),
                                           int(self.height)),
                                          flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_REPLICATE))
                if img_c == 0:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video = cv2.VideoWriter(
                                  op.join(str(pathout), str(videofilename)),
                                  fourcc, float(self.fps),
                                  (int(self.width), int(self.height)))

                video.write(img[img_c])
                img_c = img_c + 1
            else:
                break

        self.cap.release()
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

    def grayscale_video(self, pathout):

        '''
        videofilename = '/home/user/e3v810a-20190307T0740-0840.mp4'
        lstream = 0
        lstream is 1 is video is streaming or 0 if video is already saved
        v  = NTKVideos(videofilename, lstream)
        v contains length, width, height information from video

        grayscale_video
        outpath = '/home/user/out/', where to save new files
        v.grayscale_video(outpath)
        '''

        import os.path as op

        # img = []
        img_c = 0
        videofilename = (op.splitext(self.name)[0] + str("_r") +
                         op.splitext(self.name)[1])
        while True:
            # Capture frame-by-frame
            self.ret, self.frame = self.cap.read()

            if self.ret is True:
                gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                if img_c == 0:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video = cv2.VideoWriter(
                                  op.join(str(pathout), str(videofilename)),
                                  fourcc, float(self.fps),
                                  (int(self.width), int(self.height)),
                                  isColor=False)

                video.write(gray_frame)
                img_c = img_c + 1
            else:
                break

        # self.cap.release()
        video.release()
        # cv2.destroyAllWindows()

    def graydiff_video(self, pathout):

        '''
        videofilename = '/home/user/e3v810a-20190307T0740-0840.mp4'
        lstream = 0
        lstream is 1 is video is streaming or 0 if video is already saved
        v  = NTKVideos(videofilename, lstream)
        v contains length, width, height information from video

        diff_video
        outpath = '/home/user/out/', where to save new files
        v.graydiff_video(outpath)
        '''

        import os.path as op

        threshold_value = 40
        set_to_value = 255
        gray_frame_prev = 0
        # img = []
        img_c = 0
        videofilename = (op.splitext(self.name)[0] + str("_r") +
                         op.splitext(self.name)[1])
        while True:
            # Capture frame-by-frame
            self.ret, self.frame = self.cap.read()

            if self.ret is True:
                gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                # print(gray_frame.shape)
                # Take the pixel-by-pixel absolute difference of the two images
                if img_c != 0:
                    diff = cv2.absdiff(gray_frame, gray_frame_prev)
                    # Set every pixel that changed by 40 to 255
                    # and all others to zero.
                    ret, gray_framed = cv2.threshold(diff, threshold_value,
                                                     set_to_value,
                                                     cv2.THRESH_BINARY)
                if img_c == 0:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video = cv2.VideoWriter(
                                  op.join(str(pathout), str(videofilename)),
                                  fourcc, float(self.fps),
                                  (int(self.width), int(self.height)),
                                  isColor=False)

                if img_c != 0:
                    video.write(gray_framed)
                img_c = img_c + 1
                gray_frame_prev = gray_frame
            else:
                break

        # self.cap.release()
        video.release()
        # cv2.destroyAllWindows()

    def graydiff_img(self, pathout):

        '''
        videofilename = '/home/user/e3v810a-20190307T0740-0840.mp4'
        lstream = 0
        lstream is 1 is video is streaming or 0 if video is already saved
        v  = NTKVideos(videofilename, lstream)
        v contains length, width, height information from video

        diff_video
        outpath = '/home/user/out/', where to save new files
        v.graydiff_img(outpath)
        '''

        # import os.path as op

        # threshold_value = 40
        # set_to_value = 255
        gray_frame_prev = 0
        frame_num = 1
        # img = []
        img_c = 0
        while True:
            # Capture frame-by-frame
            self.ret, self.frame = self.cap.read()

            if self.ret is True:
                gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                # print(gray_frame.shape)
                # Take the pixel-by-pixel absolute difference of the two images
                if img_c != 0:
                    diff = cv2.absdiff(gray_frame, gray_frame_prev)
                    # Set every pixel that changed by 40 to 255
                    # and all others to zero.
                    # ret, gray_framed = cv2.threshold(diff, threshold_value,
                    # set_to_value, cv2.THRESH_BINARY)

                if img_c == 0:
                    gray_frame_prev = gray_frame
                elif img_c != 0:
                    gray_frame_prev = diff
                img_c = img_c + 1
            else:
                break

        cv2.imwrite(os.path.join(pathout, "frame%d.jpg") %
                    frame_num, gray_frame_prev)
        # self.cap.release()
        # video.release()
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


def get_video_length_list(videodir, lstream=0):

    '''
    Read video files and return list of all video lengths

    get_video_length_list(videodir, lstream):

    Parameters
    ----------
    videodir : directory of videos
    lstream : 1 is video is streaming or 0 (defualt) if video is already saved

    Returns
    -------
    video_length_list : list of all video lengths

    Raises
    ------

    See Also
    --------

    Notes
    -----

    Examples
    --------
    vl = ntk.get_video_length_list('/home/kbn/watchtower_current/data/',0)

    '''

    from neuraltoolkit import ntk_videos as ntkv
    import numpy as np
    import os
    import re

    video_length_list = []
    files = \
        np.sort([f for f in os.listdir(videodir) if re.match(r'.*\.mp4', f)])
    print("length of files ", len(files))
    for f in files:
        if f.endswith(".mp4"):
            # print(f)
            filename_with_path = os.path.join(videodir, f)
            # print(filename_with_path)
            v = ntkv.NTKVideos(filename_with_path, lstream)
            print(v.length)
            video_length_list.append(v.length)

    return video_length_list


if __name__ == '__main__':
    videofilename = 'e3v810a-20190307T0740-0840.mp4'
    # videofilename = 0
    lstream = 0
    v = NTKVideos(videofilename, lstream)
