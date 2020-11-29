#include "kmeans.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

template<typename T>
std::pair<cv::Mat, cv::Mat> compute_kmeans_segmantation(cv::Mat src, int K)
{
	std::vector<T> samples;

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			T value = src.at<T>(i, j);
			samples.push_back(value);
		}
	}

	auto results = kmeans::Cluster<T>(samples.cbegin(), samples.cend(), K);

	auto centorids = results.first;
	auto sample_ids = results.second;

	cv::Mat ids;
	ids.create(src.size(), CV_8U);

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			ids.at<uint8_t>(i, j) = sample_ids[i * src.cols + j];
		}
	}

	cv::Mat dst;
	dst.create(src.size(), src.type());

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<T>(i, j) = centorids[ids.at<uint8_t>(i, j)];
		}
	}
	cv::Mat dst_rgb;
	cv::applyColorMap(ids * (255 / K), dst_rgb, cv::COLORMAP_JET);

	cv::Mat dst_conturs;
	dst_conturs.create(src.size(), src.type());

	cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(2, 2));

	cv::morphologyEx(ids, dst_conturs, cv::MorphTypes::MORPH_GRADIENT, kernel);

	dst_conturs = 1 - dst_conturs;
	dst_conturs.convertTo(dst_conturs, dst_rgb.type());
	cv::Mat dst_rgb2;
	cv::bitwise_and(dst_rgb, dst_rgb, dst_rgb2, dst_conturs);

	return std::make_pair(dst, dst_rgb2);
}


int main(int argc, char** argv)
{
	const char* filename = "../imgs/296059.jpg";

    cv::Mat src = cv::imread(filename, cv::IMREAD_COLOR);

	std::pair<cv::Mat, cv::Mat> dst_rgb = compute_kmeans_segmantation<cv::Vec3b>(src, 4);

	cv::imwrite("kmeans4_rgb_2__.png", dst_rgb.first);
	cv::imwrite("kmeans4_rgb_c_2__.png", dst_rgb.second);

	cv::Mat src_lab;
	cv::cvtColor(src, src_lab, cv::COLOR_BGR2Lab);

	cv::Mat src_channels[3];
	cv::split(src_lab, src_channels);
	cv::Mat src_ab;
	cv::merge(&src_channels[1], 2, src_ab);

	std::pair<cv::Mat, cv::Mat> dst_ab = compute_kmeans_segmantation<cv::Vec2b>(src_ab, 4);

	cv::Mat mat[2] = {127 * cv::Mat::ones(src_ab.rows, src_ab.cols, CV_8U), dst_ab.first};
	cv::Mat dst_lab;
	cv::merge(mat, 2, dst_lab);
	cv::Mat dst;
	cv::cvtColor(dst_lab, dst, cv::COLOR_Lab2BGR);

	cv::imwrite("kmeans4_lab_2__.png", dst);
	cv::imwrite("kmeans4_lab_c_2__.png", dst_ab.second);

	return 0;
}
