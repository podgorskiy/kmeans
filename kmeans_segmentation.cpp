#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "kmeans.h"
#include <algorithm>
#include <vector>

int rand(int a, int b)
{
	return a  + (b - a) * (float(rand() & 0xFFFF) /0xFFFF);
}

template<>
std::pair<uint8_t, uint8_t> GetBounds(const uint8_t* samples, int count)
{
	uint8_t min = 255;
	uint8_t max = 0;
	for (const uint8_t* p = samples; p != samples + count; ++p)
	{
		min = std::min(min, *p);
		max = std::max(min, *p);
	}
	return std::make_pair(min, max);
}

template<>
std::pair<cv::Vec3b , cv::Vec3b > GetBounds(const cv::Vec3b* samples, int count)
{
	std::vector<uint8_t> samples0;
	std::vector<uint8_t> samples1;
	std::vector<uint8_t> samples2;
	for (const cv::Vec3b* sample = samples; sample != samples + count; ++sample)
	{
		samples0.push_back((*sample)[0]);
		samples1.push_back((*sample)[1]);
		samples2.push_back((*sample)[2]);
	}
	auto ch0 = GetBounds(samples0.data(), samples0.size());
	auto ch1 = GetBounds(samples1.data(), samples1.size());
	auto ch2 = GetBounds(samples2.data(), samples2.size());
	return std::make_pair(cv::Vec3b(ch0.first, ch1.first, ch2.first), cv::Vec3b(ch0.second, ch1.second, ch2.second));
}

//template<>
//cv::Vec2b GetBounds(const std::pair<T, T>& bounds)
//{
//	std::vector<uint8_t> samples0;
//	std::vector<uint8_t> samples1;
//	for (auto s: samples)
//	{
//		samples0.push_back(s[0]);
//		samples1.push_back(s[1]);
//	}
//	return cv::Vec2b(GenerateRandomSample(samples0), GenerateRandomSample(samples1));
//}
//
//template<>
//uint8_t GenerateRandomSample(const std::pair<T, T>& bounds)
//{
//	uint8_t min = *std::min_element(samples.begin(), samples.end(), [](uint8_t a, uint8_t b){ return a < b;});
//	uint8_t max = *std::max_element(samples.begin(), samples.end(), [](uint8_t a, uint8_t b){ return a < b;});
//	return rand(min, max);
//}
//
//template<>
//cv::Vec3b GenerateRandomSample(const std::pair<T, T>& bounds)
//{
//	std::vector<uint8_t> samples0;
//	std::vector<uint8_t> samples1;
//	std::vector<uint8_t> samples2;
//	for (auto s: samples)
//	{
//		samples0.push_back(s[0]);
//		samples1.push_back(s[1]);
//		samples2.push_back(s[2]);
//	}
//	return cv::Vec3b(GenerateRandomSample(samples0), GenerateRandomSample(samples1), GenerateRandomSample(samples2));
//}
//
//template<>
//cv::Vec2b GenerateRandomSample(const std::pair<T, T>& bounds)
//{
//	std::vector<uint8_t> samples0;
//	std::vector<uint8_t> samples1;
//	for (auto s: samples)
//	{
//		samples0.push_back(s[0]);
//		samples1.push_back(s[1]);
//	}
//	return cv::Vec2b(GenerateRandomSample(samples0), GenerateRandomSample(samples1));
//}

template<>
float ComputeDistance(const uint8_t& a, const uint8_t& b)
{
	float d = float(a) - float(b);
	return d * d;
}

template<>
float ComputeDistance(const cv::Vec3b& a, const cv::Vec3b& b)
{
	cv::Vec3f p = cv::Vec3f(a) - cv::Vec3f(b);
	return p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
}

template<>
float ComputeDistance(const cv::Vec2b& a, const cv::Vec2b& b)
{
	cv::Vec2f p = cv::Vec2f(a) - cv::Vec2f(b);
	return p[0] * p[0] + p[1] * p[1];
}

//template<>
//uint8_t ComputeMean(const std::vector<uint8_t>& samples)
//{
//	float s = 0;
//	for (auto& x: samples)
//	{
//		s += x;
//	}
//	return uint8_t(s / samples.size());
//}
//
//template<>
//cv::Vec3b ComputeMean(const std::vector<cv::Vec3b>& samples)
//{
//	cv::Vec3f s = 0;
//	for (auto& x: samples)
//	{
//		s = s + cv::Vec3f(x);
//	}
//	float k = 1.0f / samples.size();
//	return cv::Vec3b(s.mul(cv::Vec3f(k, k, k)));
//}
//
//template<>
//cv::Vec2b ComputeMean(const std::vector<cv::Vec2b>& samples)
//{
//	cv::Vec2f s = 0;
//	for (auto& x: samples)
//	{
//		s = s + cv::Vec2f(x);
//	}
//	float k = 1.0f / samples.size();
//	return cv::Vec2b(s.mul(cv::Vec2f(k, k)));
//}

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

	auto results = Kmeans(samples, K);

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
	cv::applyColorMap(ids * (255.0f / K), dst_rgb, cv::COLORMAP_JET);

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
	const char* filename = "296059.jpg";
	if (argc == 2)
	{
		filename = argv[1];
	}
    cv::Mat src = cv::imread(filename, cv::IMREAD_COLOR);
	cv::Mat src_gray;
	cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

	std::pair<cv::Mat, cv::Mat> dst_gray = compute_kmeans_segmantation<uint8_t>(src_gray, 2);

	cv::imwrite("kmeans4_gray_2.png", dst_gray.first);
	cv::imwrite("kmeans4_gray_c_2.png", dst_gray.second);

	std::pair<cv::Mat, cv::Mat> dst_rgb = compute_kmeans_segmantation<cv::Vec3b>(src, 2);

	cv::imwrite("kmeans4_rgb_2.png", dst_rgb.first);
	cv::imwrite("kmeans4_rgb_c_2.png", dst_rgb.second);

	cv::Mat src_lab;
	cv::cvtColor(src, src_lab, cv::COLOR_BGR2Lab);

	cv::Mat src_channels[3];
	cv::split(src_lab, src_channels);
	cv::Mat src_ab;
	cv::merge(&src_channels[1], 2, src_ab);

	std::pair<cv::Mat, cv::Mat> dst_ab = compute_kmeans_segmantation<cv::Vec2b>(src_ab, 2);

	cv::Mat mat[2] = {127 * cv::Mat::ones(src_ab.rows, src_ab.cols, CV_8U), dst_ab.first};
	cv::Mat dst_lab;
	cv::merge(mat, 2, dst_lab);
	cv::Mat dst;
	cv::cvtColor(dst_lab, dst, cv::COLOR_Lab2BGR);

	cv::imwrite("kmeans4_lab_2.png", dst);
	cv::imwrite("kmeans4_lab_c_2.png", dst_ab.second);

	return 0;
}
