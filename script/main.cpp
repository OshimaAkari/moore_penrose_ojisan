/**
 * @file main.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-07-09
 *
 * @copyright Copyright (c) 2023
 *
 */

// iostream
#include <iostream>

// vector
#include <vector>

// opencv
#include "opencv2/opencv.hpp"

// dlib
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

//! 顔検出器
const std::string FACE_DETECTOR = "/home/oshima/workspace/moore_penrose_ojisan/config/shape_predictor_68_face_landmarks.dat";
//! 顔
const std::string FACE_IMAGE = "/home/oshima/workspace/moore_penrose_ojisan/resource/face.png";
//! 目
const std::string EYE_IMAGE = "/home/oshima/workspace/moore_penrose_ojisan/resource/eye.png";
//! 閉じてる目の画像
const std::string EYE_CLOSE_IMAGE = "/home/oshima/workspace/moore_penrose_ojisan/resource/eye_close.png";
//! 顔の回転中心
const cv::Point FACE_CENTER = cv::Point(200, 400);

//! リサイズ割合
const float RESIZE_RATE = 0.3;

//! 目の最大の大きさ
const float MAX_EYE_PIXEL = 6;

//! 目のつぶるしきい値
const float EYE_CLOSE_THRESHOLD = 3;

//! 左目の位置
const cv::Point LEFT_EYE_POSITION = cv::Point(100, 220);
//! 右目の位置
const cv::Point2f RIGHT_EYE_POSITION = cv::Point2f(230, 220);

int main(int, char **)
{
    cv::Mat frame;
    cv::VideoCapture cap(0);
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    cv::Mat face_image = cv::imread(FACE_IMAGE, cv::IMREAD_COLOR);
    cv::Mat eye_image = cv::imread(EYE_IMAGE, cv::IMREAD_UNCHANGED);
    cv::Mat close_eye_image = cv::imread(EYE_CLOSE_IMAGE, cv::IMREAD_UNCHANGED);

    std::vector<cv::Mat> split_map;
    cv::split(eye_image, split_map);
    std::vector<cv::Mat> color_map = {split_map[0], split_map[1], split_map[2]};
    std::vector<cv::Mat> mask_map = {split_map[3]};
    cv::Mat eye_color, eye_mask;
    cv::merge(color_map, eye_color);
    cv::merge(mask_map, eye_mask);

    cv::split(close_eye_image, split_map);
    color_map = {split_map[0], split_map[1], split_map[2]};
    mask_map = {split_map[3]};
    cv::Mat eye_closed_color, eye_closed_mask;
    cv::merge(color_map, eye_closed_color);
    cv::merge(mask_map, eye_closed_mask);

    cv::Mat canvas = face_image.clone();

    cv::imshow("face", canvas);

    dlib::shape_predictor sp;
    dlib::deserialize(FACE_DETECTOR) >> sp;

    int key;

    while (cap.read(frame))
    {

        cv::Mat resize_mat;
        cv::resize(frame, resize_mat, cv::Size(int(frame.cols * RESIZE_RATE), int(frame.rows * RESIZE_RATE)));

        dlib::cv_image<dlib::bgr_pixel> dlib_img(resize_mat);
        std::vector<dlib::rectangle> dets = detector(dlib_img);

        if (dets.size() > 0)
        {
            canvas = face_image.clone();
        }

        for (unsigned long j = 0; j < dets.size(); ++j)
        {

            dlib::full_object_detection shape = sp(dlib_img, dets[j]);

            // 顔の傾きを算出
            auto point1 = shape.part(30);
            auto point2 = shape.part(27);
            double angle = std::atan2(-(point1.y() - point2.y()), point1.x() - point2.x());
            angle = angle / M_PI * 180 + 90;

            point1 = shape.part(38);
            point2 = shape.part(40);
            auto left_dist = std::sqrt(std::pow(point1.x() - point2.x(), 2) + std::pow(point1.y() - point2.y(), 2));

            if (left_dist < EYE_CLOSE_THRESHOLD)
            {
                //! 閉じている目をコピー
                for (int v = 0; v < eye_closed_color.rows; v++)
                {
                    int project_v = v + LEFT_EYE_POSITION.y;
                    if (project_v > face_image.rows - 1)
                    {
                        continue;
                    }

                    cv::Vec3b *eye_closed_color_pix = eye_closed_color.ptr<cv::Vec3b>(v);
                    cv::Vec3b *face_image_pix = canvas.ptr<cv::Vec3b>(project_v);
                    unsigned char *eye_closed_mask_pix = eye_closed_mask.ptr<unsigned char>(v);

                    for (int u = 0; u < eye_closed_color.cols; u++)
                    {
                        int project_u = u + LEFT_EYE_POSITION.x;
                        if (project_u > face_image.cols - 1)
                        {
                            continue;
                        }

                        if (eye_closed_mask_pix[u] == 0)
                        {
                            continue;
                        }
                        face_image_pix[project_u] = eye_closed_color_pix[u];
                    }
                }
            }
            else
            {
                // 開いている目をコピー
                for (int v = 0; v < eye_color.rows; v++)
                {
                    int project_v = v + LEFT_EYE_POSITION.y;
                    if (project_v > face_image.rows - 1)
                    {
                        continue;
                    }

                    cv::Vec3b *eye_color_pix = eye_color.ptr<cv::Vec3b>(v);
                    cv::Vec3b *face_image_pix = canvas.ptr<cv::Vec3b>(project_v);
                    unsigned char *eye_mask_pix = eye_mask.ptr<unsigned char>(v);

                    for (int u = 0; u < eye_color.cols; u++)
                    {
                        int project_u = u + LEFT_EYE_POSITION.x;
                        if (project_u > face_image.cols - 1)
                        {
                            continue;
                        }

                        if (eye_mask_pix[u] == 0)
                        {
                            continue;
                        }
                        face_image_pix[project_u] = eye_color_pix[u];
                    }
                }
            }

            point1 = shape.part(44);
            point2 = shape.part(46);
            auto right_dist = std::sqrt(std::pow(point1.x() - point2.x(), 2) + std::pow(point1.y() - point2.y(), 2));

            if (right_dist < EYE_CLOSE_THRESHOLD)
            {
                //! 閉じている目をコピー
                for (int v = 0; v < eye_closed_color.rows; v++)
                {
                    int project_v = v + RIGHT_EYE_POSITION.y;
                    if (project_v > face_image.rows - 1)
                    {
                        continue;
                    }

                    cv::Vec3b *eye_closed_color_pix = eye_closed_color.ptr<cv::Vec3b>(v);
                    cv::Vec3b *face_image_pix = canvas.ptr<cv::Vec3b>(project_v);
                    unsigned char *eye_closed_mask_pix = eye_closed_mask.ptr<unsigned char>(v);

                    for (int u = 0; u < eye_closed_color.cols; u++)
                    {
                        int project_u = u + RIGHT_EYE_POSITION.x;
                        if (project_u > face_image.cols - 1)
                        {
                            continue;
                        }

                        if (eye_closed_mask_pix[u] == 0)
                        {
                            continue;
                        }
                        face_image_pix[project_u] = eye_closed_color_pix[u];
                    }
                }
            }
            else
            {
                // 開いている目をコピー
                for (int v = 0; v < eye_color.rows; v++)
                {
                    int project_v = v + RIGHT_EYE_POSITION.y;
                    if (project_v > face_image.rows - 1)
                    {
                        continue;
                    }

                    cv::Vec3b *eye_color_pix = eye_color.ptr<cv::Vec3b>(v);
                    cv::Vec3b *face_image_pix = canvas.ptr<cv::Vec3b>(project_v);
                    unsigned char *eye_mask_pix = eye_mask.ptr<unsigned char>(v);

                    for (int u = 0; u < eye_color.cols; u++)
                    {
                        int project_u = u + RIGHT_EYE_POSITION.x;
                        if (project_u > face_image.cols - 1)
                        {
                            continue;
                        }

                        if (eye_mask_pix[u] == 0)
                        {
                            continue;
                        }
                        face_image_pix[project_u] = eye_color_pix[u];
                    }
                }
            }

            // 顔を回転させる
            auto rot_mat = cv::getRotationMatrix2D(FACE_CENTER, angle, 1.0);
            cv::warpAffine(canvas, canvas, rot_mat, canvas.size(), 1, 0, cv::Scalar(255, 255, 255));
        }

        cv::imshow("test", resize_mat);
        cv::imshow("face", canvas);

        key = cv::waitKey(1);

        if (key == 27)
        {
            break;
        }
    }

    std::cout << "Process Done!\n";
    return 0;
}
