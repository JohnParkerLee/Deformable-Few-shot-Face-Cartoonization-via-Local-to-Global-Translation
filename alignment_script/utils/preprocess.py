from .face_detect import FaceDetect
from .face_seg import FaceSeg
import numpy as np


class Preprocess:
    def __init__(self, device='cpu', detector='sfd'):
        self.detect = FaceDetect(device, detector)  # device = 'cpu' or 'cuda', detector = 'dlib' or 'sfd'
        self.segment = FaceSeg()

    def face_landmark_info(self, image):
        return self.detect.get_face_landmarks(image)

    def process(self, image, landmarks):
        face_info = self.detect.align(image, landmarks)
        if face_info is None:
            return None
        image_align, landmarks_align = face_info
        # crop the needed img
        face, landmarks = self.__crop(image_align, landmarks_align)
        # print('points')
        # mask = self.segment.get_mask(face)
        # return np.dstack((face, mask))
        # return image_align, landmarks_align
        # print(face.shape)
        return face, landmarks

    def segmentation(self, image):
        mask = self.segment.get_mask(image)
        face_rgba = np.dstack((image, mask))
        if face_rgba is not None:
            face = face_rgba[:,:,:3].copy()
            mask = face_rgba[:,:,3].copy()[:,:,np.newaxis]/255.
            face_white_bg = (face*mask + (1-mask)*255).astype(np.uint8)
            return face_white_bg
        return image


    @staticmethod
    def __crop(image, landmarks):
        landmarks_top = np.min(landmarks[:, 1])
        landmarks_bottom = np.max(landmarks[:, 1])
        landmarks_left = np.min(landmarks[:, 0])
        landmarks_right = np.max(landmarks[:, 0])

        # expand bbox
        top = int(landmarks_top - 0.8 * (landmarks_bottom - landmarks_top))
        bottom = int(landmarks_bottom + 0.3 * (landmarks_bottom - landmarks_top))
        left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left))
        right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left))

        if bottom - top > right - left:
            left -= ((bottom - top) - (right - left)) // 2
            right = left + (bottom - top)
        else:
            top -= ((right - left) - (bottom - top)) // 2
            bottom = top + (right - left)
        if (bottom - top + 1 > 512):
            gap = int((bottom - top - 512)/2)
            bottom -= gap
            top += gap
            right -= gap
            left += gap
            if (bottom - top + 1 != 512):
                top -= 512 - (bottom - top + 1)
                left -= 512 - (right - left + 1)
        elif(bottom - top + 1 < 512):
            gap = int((512 - (bottom - top)) / 2)
            bottom += gap
            top -= gap
            left -= gap
            right += gap
            if (bottom - top + 1 != 512):
                top -= 512 - (bottom - top + 1)
                left -= 512 - (right - left + 1)
        if len(image.shape)==3:
            image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255
        else:
            image_crop = np.ones((bottom - top + 1, right - left + 1), np.uint8) * 255
        landmarks[:, 1] = landmarks[:, 1] - top
        landmarks[:, 0] = landmarks[:, 0] - left
        h, w = image.shape[:2]
        left_white = max(0, -left)
        left = max(0, left)
        right = min(right, w-1)
        right_white = left_white + (right-left)
        top_white = max(0, -top)
        top = max(0, top)
        bottom = min(bottom, h-1)
        bottom_white = top_white + (bottom - top)

        image_crop[top_white:bottom_white+1, left_white:right_white+1] = image[top:bottom+1, left:right+1].copy()

        return image_crop, landmarks
