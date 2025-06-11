import base64
import io
import os
import json
import numpy as np
import os.path as osp
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import cv2

__version__ = "4.5.7"
class LabelFileError(Exception):
    pass
class labelJson:
    def __init__(self):
        return
    def save_label_json(self,image, imagePath, points, filename):
        self.save_landmark_json(image, imagePath, points, filename)
        return None

    def img_data_to_pil(self,img_data):
        f = io.BytesIO()
        f.write(img_data)
        img_pil = PIL.Image.open(f)
        return img_pil

    def load(self, filename):
        keys = [
            "version",
            "imageData",
            "imagePath",
            "shapes",  # polygonal annotations
            "flags",  # image level flags
            "imageHeight",
            "imageWidth",
        ]
        shape_keys = [
            "label",
            "points",
            "group_id",
            "shape_type",
            "flags",
        ]
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            version = data.get("version")
            # if version is None:
            #     logger.warn(
            #         "Loading JSON file ({}) of unknown version".format(
            #             filename
            #         )
            #     )
            # elif version.split(".")[0] != __version__.split(".")[0]:
            #     logger.warn(
            #         "This JSON file ({}) may be incompatible with "
            #         "current labelme. version in file: {}, "
            #         "current version: {}".format(
            #             filename, version, __version__
            #         )
            #     )L

            if data["imageData"] is not None:
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                imageDecode = base64.b64decode(data["imageData"])
                imageData = self.img_data_to_pil(imageDecode)
                # if PY2 and QT4:
                #     imageData = utils.img_data_to_png_data(imageData)
            else:
                # relative path from label file to relative path from cwd
                imagePath = osp.join(osp.dirname(filename), data["imagePath"])
                imageData = self.load_image_file(imagePath)
            flags = data.get("flags") or {}
            imagePath = data["imagePath"]
            # self._check_image_height_and_width(
            #     base64.b64encode(imageData).decode("utf-8"),
            #     data.get("imageHeight"),
            #     data.get("imageWidth"),
            # )
            shapes = [
                dict(
                    label=s["label"],
                    points=s["points"],
                    shape_type=s.get("shape_type", "polygon"),
                    flags=s.get("flags", {}),
                    group_id=s.get("group_id"),
                    other_data={
                        k: v for k, v in s.items() if k not in shape_keys
                    },
                )
                for s in data["shapes"]
            ]
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.filename = filename
        self.otherData = otherData

    def img_arr_to_b64(self,image):
        img_pil = PIL.Image.fromarray(image)
        f = io.BytesIO()
        img_pil.save(f, format="PNG")
        img_bin = f.getvalue()
        if hasattr(base64, "encodebytes"):
            img_b64 = base64.encodebytes(img_bin)
        else:
            img_b64 = base64.encodestring(img_bin)
        return img_b64

    def get_image_height_and_width(self,image):
        return image.height, image.width

    def get_bytes_img(self, image, filename):
        # filename = '骑士2-01.jpg'
        # image = PIL.Image.open('C:\\Users\\Shengshu\\Desktop\\2-01.jpg')
        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            else:
                format = "PNG"
            image.save(f, format=format)
            f.seek(0)
            return f.read()

    # input: image, imagePath, points
    # output: landmark.json
    def save_landmark_json(self, image, imagePath, points, filename, save_path):
        # shapes = []
        # for i in range(len(points)):
        #     shapes.append(
        #         dict(
        #             label=str(i),
        #             points=[points[i]],
        #             shape_type="point",
        #             flags={},
        #             group_id=None,
        #         )
        #     )
        shapes = [dict(
                    label="1",
                    points=points,
                    shape_type="point",
                    flags={},
                    group_id=None,
                )
        ]
        imageHeight, imageWidth =self.get_image_height_and_width(image)
        # image = load_image_data(filename)
        # imageData = img_arr_to_b64(imageData)
        ext = osp.splitext(filename)[0].lower()
        # opencv2
        imgbytes = self.get_bytes_img(image, filename)
        # imgbytes = cv2.imencode(ext, image)[1]
        imageData = base64.b64encode(imgbytes).decode("utf-8")
        flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        # print('asd')
        try:
            with open(os.path.join(save_path,ext+'.json'), "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                # print('finish')
        except Exception as e:
            raise LabelFileError(e)

    def load_image_data(self,filename):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            # logger.error("Failed opening image file: {}".format(filename))
            return
        # apply orientation to image according to exif
        image_pil = self.apply_exif_orientation(image_pil)
        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            else:
                format = "PNG"
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def apply_exif_orientation(self,image):
        try:
            exif = image._getexif()
        except AttributeError:
            exif = None

        if exif is None:
            return image

        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in exif.items()
            if k in PIL.ExifTags.TAGS
        }

        orientation = exif.get("Orientation", None)

        if orientation == 1:
            # do nothing
            return image
        elif orientation == 2:
            # left-to-right mirror
            return PIL.ImageOps.mirror(image)
        elif orientation == 3:
            # rotate 180
            return image.transpose(PIL.Image.ROTATE_180)
        elif orientation == 4:
            # top-to-bottom mirror
            return PIL.ImageOps.flip(image)
        elif orientation == 5:
            # top-to-left mirror
            return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
        elif orientation == 6:
            # rotate 270
            return image.transpose(PIL.Image.ROTATE_270)
        elif orientation == 7:
            # top-to-right mirror
            return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
        elif orientation == 8:
            # rotate 90
            return image.transpose(PIL.Image.ROTATE_90)
        else:
            return image