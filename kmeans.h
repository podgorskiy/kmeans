#include <stdio.h>
#include <vector>
#include <algorithm>
#include <limits>


template<typename T>
T GenerateRandomSample(const std::pair<T, T>& bounds);

template<typename T>
std::pair<T, T> GetBounds(const T* samples, int count);

template<typename T>
float ComputeDistance(const T& a, const T& b);

template<typename T>
T ComputeMean(const T* samples, int count);

template <typename T>
int PickNearestCluster(const std::vector<T>& centroids, const T& sample, float& distance)
{
	float minimumD = std::numeric_limits<float>::max();
	int nearestCluster = -1;
	for (int i = 0, l = centroids.size(); i < l; ++i)
	{
		float d = ComputeDistance(centroids[i], sample);
		if (minimumD > d)
		{
			nearestCluster = i;
			minimumD = d;
		}
	}
	distance = minimumD;
	return nearestCluster;
}

template <typename T>
std::pair<std::vector<T>, std::vector<int> > Kmeans(const T* samples, int count, int clusterCount)
{
	std::vector<T> centroids;

	std::pair<T, T> bounds = GetBounds<T>(samples, count);
	for (int i = 0; i < clusterCount; ++i)
	{
		centroids.push_back(GenerateRandomSample(bounds));
	}

	int amountOfChangedAssignments = 1;
	int step = 0;

	std::vector<int> sample_ids;
	sample_ids.resize(count);

	while (amountOfChangedAssignments != 0)
	{
		// Assigning every item to it's nearest cluster center
		amountOfChangedAssignments = 0;
		float objective = 0;
		for (int i = 0; i < count; ++i)
		{
			float d;
			int nearestCluster = PickNearestCluster(centroids, samples[i], d);
			objective += d;
			if (nearestCluster != sample_ids[i])
			{
				amountOfChangedAssignments++;
				sample_ids[i] = nearestCluster;
			}
		}
		printf("%d %d %f\n", step, amountOfChangedAssignments, objective);
		step++;

		for (int i = 0; i < clusterCount; ++i)
		{
			std::vector<T> samples_in_cluster;

			for (int j = 0; j < count; ++j)
			{
				if (sample_ids[j] == i)
				{
					samples_in_cluster.push_back(samples[j]);
				}
			}

			centroids[i] = ComputeMean(samples_in_cluster.data(), samples_in_cluster.size());
		}
	}

	return std::make_pair(centroids, sample_ids);
}

template <typename T>
std::pair<std::vector<T>, std::vector<int> > Kmeans(const std::vector<T>& samples, int clusterCount)
{
	return Kmeans(samples.data(), samples.size(), clusterCount);
}
