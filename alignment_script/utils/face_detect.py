import cv2
import math
import numpy as np
import face_alignment
import dlib

class FaceDetect:
    def __init__(self, device, detector):
        # landmarks will be detected by face_alignment library. Set device = 'cuda' if use GPU.
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector=detector)
        self.frontalFaceDetector = dlib.get_frontal_face_detector()
        self.faceLandmarkDetector = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    def get_face_landmarks(self,image):
        ### 还需要判断空的情况
        allFaces = self.frontalFaceDetector(image, 0)
        for k in range(0, len(allFaces)):
            # dlib rectangle class will detecting face so that landmark can apply inside of that area
            faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()), int(allFaces[k].top()),
                                               int(allFaces[k].right()), int(allFaces[k].bottom()))

            # Now we are running loop on every detected face and putting landmark on that with the help of faceLandmarkDetector
            detectedLandmarks = self.faceLandmarkDetector(image, faceRectangleDlib)

            # count number of landmarks we actually detected on image
            if k == 0:
                print("Total number of face landmarks detected ", len(detectedLandmarks.parts()))

            # Svaing the landmark one by one to the output folder
            # allFacesLandmark.append(detectedLandmarks)
            #
            # # Now finally we drawing landmarks on face
            # facePoints(img, detectedLandmarks)
            #
            # fileName = faceLandmarksOuput + "_" + str(k) + ".txt"
            # print("Lanmdark is save into ", fileName)

            # Write landmarks to disk
            # writeFaceLandmarksToLocalFile(detectedLandmarks, fileName)
        # return self.fa.get_landmarks(image)
        points = []
        for i in range(0, 68):
            point = [detectedLandmarks.part(i).x, detectedLandmarks.part(i).y]
            points.append(point)

        # points = np.array(points, dtype=np.int32)
        return points

    def align(self, image, landmarks):
        # landmarks = self.__get_max_face_landmarks(image)

        if landmarks is None:
            return None
        else:
            return self.__rotate(image, landmarks)

    def __get_max_face_landmarks(self, image):
        preds = self.fa.get_landmarks(image)
        if preds is None:
            return None

        elif len(preds) == 1:
            return preds[0]

        else:
            # find max face
            areas = []
            for pred in preds:
                landmarks_top = np.min(pred[:, 1])
                landmarks_bottom = np.max(pred[:, 1])
                landmarks_left = np.min(pred[:, 0])
                landmarks_right = np.max(pred[:, 0])
                areas.append((landmarks_bottom - landmarks_top) * (landmarks_right - landmarks_left))
            max_face_index = np.argmax(areas)
            return preds[max_face_index]

    @staticmethod
    def __rotate(image, landmarks):
        # rotation angle
        left_eye_corner = landmarks[36]
        right_eye_corner = landmarks[45]
        radian = np.arctan((left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))

        # image size after rotating
        height, width = image.shape[:2]
        cos = math.cos(radian)
        sin = math.sin(radian)
        new_w = int(width * abs(cos) + height * abs(sin))
        new_h = int(width * abs(sin) + height * abs(cos))

        # translation
        Tx = new_w // 2 - width // 2
        Ty = new_h // 2 - height // 2

        # affine matrix
        M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx],
                      [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])

        image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

        landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
        landmarks_rotate = np.dot(M, landmarks.T).T
        return image_rotate, landmarks_rotate


if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread('3989161_1.jpg'), cv2.COLOR_BGR2RGB)
    fd = FaceDetect(device='cpu')
    face_info = fd.align(img)
    if face_info is not None:
        image_align, landmarks_align = face_info

        for i in range(landmarks_align.shape[0]):
            cv2.circle(image_align, (int(landmarks_align[i][0]), int(landmarks_align[i][1])), 2, (255, 0, 0), -1)

        cv2.imwrite('image_align.png', cv2.cvtColor(image_align, cv2.COLOR_RGB2BGR))
