# kmeans
Simple, one-header, drop-in, templated library for kmeans

![jpeg](imgs/296059.jpg)
![jpeg](imgs/kmeans4_rgb_2__.png)

Works out of the box with scalars and vector types from OpenCV and GLM, such as glm::vec2, glm::vec3, cv::Vec3b, etc (Euclidean metric is assumed).

For everything else, provide specialization of:

```cpp
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
}
```

Where you need to 

 * specify `DistanceType` (typically float)
 * specify function for random sampling `GenerateRandomSample`, given the bounds
 * specify function for computing bounds `GetBounds`, given the sample list
 * specify function for computing distance `ComputeDistance`, given the two samples which can be non euclidean to.
 * specify function for computing mean of given sample list `ComputeMean`.
 