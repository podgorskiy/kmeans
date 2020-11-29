//  Copyright 2020 Stanislav Pidhorskyi
//  Licensed under the MIT license:
//
//      http://www.opensource.org/licenses/mit-license.php
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.

#include <stdio.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <type_traits>


namespace kmeans
{
	// Provide your own specialization for your data type
	// Already have default ones for scalars and vector types as in cv::Vec<> and glm::vec_t<>
	template<typename T, class Enable = void>
	class DataTypeImpl
	{
	public:
		typedef float DistanceType;

		static T GenerateRandomSample(const std::pair<T, T>& bounds);

		static std::pair<T, T> GetBounds(const T* first, const T* end);

		static DistanceType ComputeDistance(const T& a, const T& b);

		static T ComputeMean(const T* first, const T* end);
	};

	template<typename T>
	inline int PickNearestCluster(const std::vector<T>& centroids, const T& sample, float& distance)
	{
		auto minimumD = DataTypeImpl<T>::ComputeDistance(centroids[0], sample);
		int nearestCluster = 0;
		for (int i = 1, l = centroids.size(); i < l; ++i)
		{
			auto d = DataTypeImpl<T>::ComputeDistance(centroids[i], sample);
			if (minimumD > d)
			{
				nearestCluster = i;
				minimumD = d;
			}
		}
		distance = minimumD;
		return nearestCluster;
	}

	template<typename T>
	inline std::pair<std::vector<T>, std::vector<int> > Cluster(const T* first, const T* last, int clusterCount)
	{
		std::vector<T> centroids;
		centroids.reserve(clusterCount);

		std::pair<T, T> bounds = DataTypeImpl<T>::GetBounds(first, last);
		for (int i = 0; i < clusterCount; ++i)
		{
			centroids.emplace_back(DataTypeImpl<T>::GenerateRandomSample(bounds));
		}

		unsigned int amountOfChangedAssignments = 1;
		int step = 0;

		std::vector<int> sample_ids;
		sample_ids.resize(last - first);

		while (amountOfChangedAssignments != 0)
		{
			// Assigning every item to it's nearest cluster center
			amountOfChangedAssignments = 0;
			float objective = 0;
			for (int i = 0; i < last - first; ++i)
			{
				float d;
				int nearestCluster = PickNearestCluster(centroids, first[i], d);
				objective += d;
				if (nearestCluster != sample_ids[i])
				{
					amountOfChangedAssignments++;
					sample_ids[i] = nearestCluster;
				}
			}
			// printf("%d %d %f\n", step, amountOfChangedAssignments, objective);
			step++;

			std::vector<T> samples_in_cluster;
			samples_in_cluster.reserve(last - first);

			for (int i = 0; i < clusterCount; ++i)
			{
				samples_in_cluster.clear();
				for (int j = 0; j < last - first; ++j)
				{
					if (sample_ids[j] == i)
					{
						samples_in_cluster.push_back(first[j]);
					}
				}

				if (samples_in_cluster.size() == 0)
				{
					auto max_distance = DataTypeImpl<T>::ComputeDistance(centroids[sample_ids[0]], first[0]);
					int max_distance_c = 0;
					for (int j = 1; j < last - first; ++j)
					{
						auto d = DataTypeImpl<T>::ComputeDistance(centroids[sample_ids[j]], first[j]);
						if (d > max_distance)
						{
							max_distance_c = j;
							max_distance = d;
						}
					}
					centroids[i] = first[max_distance_c];
					sample_ids[max_distance_c] = i;
					samples_in_cluster.push_back(first[max_distance_c]);
					amountOfChangedAssignments |= 1U;
				}

				const T* ptr = samples_in_cluster.data();
				int size = samples_in_cluster.size();
				centroids[i] = DataTypeImpl<T>::ComputeMean(ptr, ptr + size);
			}
		}

		return std::make_pair(centroids, sample_ids);
	}

	template<typename T>
	std::pair<std::vector<T>, std::vector<int> >
	inline Cluster(typename std::vector<T>::const_iterator first, typename std::vector<T>::const_iterator last,
	        int clusterCount)
	{
		const T* ptr = &*first;
		int size = last - first;
		return Cluster(ptr, ptr + size, clusterCount);
	}

	int rand(int a, int b)
	{
		return a + (b - a) * (float(std::rand() & 0xFFFF) / 0xFFFF);
	}

	template<class T, class Enable = void>
	struct make_long
	{
		typedef T type;
	};

	template<class T>
	struct make_long<T, typename std::enable_if<std::is_floating_point<T>::value>::type>
	{
		typedef double type;
	};

	template<class T>
	struct make_long<T, typename std::enable_if<std::is_integral<T>::value>::type>
	{
		typedef int64_t type;
	};

	template<typename Iterator>
	inline Iterator max_element(Iterator first, Iterator last, int stride)
	{
		if (first == last) return first;
		Iterator result = first;
		for (; first != last; first += stride)
			if (result < first)
				result = first;
		return result;
	}

	template<typename Iterator>
	inline Iterator min_element(Iterator first, Iterator last, int stride)
	{
		if (first == last) return first;
		Iterator result = first;
		for (; first != last; first += stride)
			if (result > first)
				result = first;
		return result;
	}

	template<typename T, template<typename, int> class Vec>
	class DataTypeImpl<Vec<T, 2> >
	{
	public:
		typedef Vec<T, 2> vec2_type;
		typedef typename make_long<typename std::make_signed<T>::type>::type DistanceType;
		typedef Vec<DistanceType, 2> svec2_type;
		typedef Vec<typename make_long<T>::type, 2> lvec2_type;

		static vec2_type ComputeMean(const vec2_type* first, const vec2_type* last)
		{
			auto size = last - first;
			if (size == 0)
				return vec2_type(0, 0);
			lvec2_type s = 0;
			for (; first != last; ++first)
			{
				s = s + lvec2_type(*first);
			}
			return vec2_type((s[0] + size / 2) / size, (s[1] + size / 2) / size);
		}

		static DistanceType ComputeDistance(const vec2_type& a, const vec2_type& b)
		{
			svec2_type p = svec2_type(a) - svec2_type(b);
			return p[0] * p[0] + p[1] * p[1];
		}

		static vec2_type GenerateRandomSample(const std::pair<vec2_type, vec2_type>& bounds)
		{
			return vec2_type(rand(bounds.first[0], bounds.second[0]), rand(bounds.first[1], bounds.second[1]));
		}

		static std::pair<vec2_type, vec2_type> GetBounds(const vec2_type* first, const vec2_type* last)
		{
			const typename vec2_type::value_type* data = first->val;
			const typename vec2_type::value_type* end = last->val;
			return std::make_pair(
					vec2_type(
							*min_element(data + 0, end + 0, 2),
							*min_element(data + 1, end + 1, 2)),
					vec2_type(
							*max_element(data + 0, end + 0, 2),
							*max_element(data + 1, end + 1, 2)));
		}
	};

	template<typename T, template<typename, int> class Vec>
	class DataTypeImpl<Vec<T, 3> >
	{
	public:
		typedef Vec<T, 3> vec3_type;
		typedef typename make_long<typename std::make_signed<T>::type>::type DistanceType;
		typedef Vec<DistanceType, 3> svec3_type;
		typedef Vec<typename make_long<T>::type, 3> lvec3_type;

		static vec3_type ComputeMean(const vec3_type* first, const vec3_type* last)
		{
			auto size = last - first;
			if (size == 0)
				return vec3_type(0, 0, 0);
			lvec3_type s = 0;
			for (; first != last; ++first)
			{
				s = s + lvec3_type(*first);
			}
			return vec3_type((s[0] + size / 2) / size, (s[1] + size / 2) / size, (s[2] + size / 2) / size);
		}

		static DistanceType ComputeDistance(const vec3_type& a, const vec3_type& b)
		{
			svec3_type p = svec3_type(a) - svec3_type(b);
			return p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
		}

		static vec3_type GenerateRandomSample(const std::pair<vec3_type, vec3_type>& bounds)
		{
			return vec3_type(rand(bounds.first[0], bounds.second[0]), rand(bounds.first[1], bounds.second[1]),
			                 rand(bounds.first[2], bounds.second[2]));
		}

		static std::pair<vec3_type, vec3_type> GetBounds(const vec3_type* first, const vec3_type* last)
		{
			const typename vec3_type::value_type* data = first->val;
			const typename vec3_type::value_type* end = last->val;
			return std::make_pair(
					vec3_type(
							*min_element(data + 0, end + 0, 3),
							*min_element(data + 1, end + 1, 3),
							*min_element(data + 2, end + 2, 3)),
					vec3_type(
							*max_element(data + 0, end + 0, 3),
							*max_element(data + 1, end + 1, 3),
							*max_element(data + 2, end + 2, 3)));
		}
	};

	template<typename T>
	class DataTypeImpl<T, typename std::enable_if<std::is_floating_point<T>::value || std::is_integral<T>::value>::type>
	{
	public:
		typedef typename make_long<typename std::make_signed<T>::type>::type DistanceType;

		inline std::pair<T, T> GetBounds(const T* first, const T* last)
		{
			T min = *std::min_element(first, last);
			T max = *std::max_element(first, last);
			return std::make_pair(min, max);
		}

		inline T GenerateRandomSample(const std::pair<T, T>& bounds)
		{
			return rand(bounds.first, bounds.second);
		}

		inline DistanceType ComputeDistance(const T& a, const T& b)
		{
			auto d = DistanceType(a) - DistanceType(b);
			return d * d;
		}

		inline T ComputeMean(const T* first, const T* last)
		{
			auto size = last - first;
			if (size == 0)
				return 0;
			make_long<T> s = 0;
			for (; first != last; ++first)
			{
				s += *first;
			}
			return (s + size / 2) / size;
		}
	};
}
