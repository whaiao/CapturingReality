// HDR.cpp : Defines the entry point for the application.
//

#include "HDR.h"

using namespace std;
using namespace cv;

// mentioned in lecture 3 at start
#define THRESHOLD 10e-4f

float robWeight(int bit8)
{
	float q = 255 / 4.f;
	float value = bit8 / q - 2.f;
	value = exp(-value * value);
	return value;
}

int findInSet(int x, std::vector<int>& fa)
{
	return fa[x] = (fa[x] == x ? x : findInSet(fa[x], fa));
}

void unio(int a, int b, std::vector<int>& fa, std::vector<int>& memCounter)
{
	if (findInSet(a, fa) != findInSet(b, fa))
		memCounter[findInSet(b, fa)] += memCounter[findInSet(a, fa)];
	fa[findInSet(a, fa)] = findInSet(b, fa);
}

// Traverse check
bool inside(int r, int c, int rows, int cols)
{
	return r >= 0 && r < rows && c >= 0 && c < cols;
}

int main()
{
	vector<float> exposure;
	vector<Mat> images;
	string topLevelDirectory = "C:/Users/benja/Development/CapturingReality/HDR/data/";
	string hdrgenFile = topLevelDirectory + "max.hdrgen";
	ifstream file(hdrgenFile);
	if (file.is_open())
	{
		string line;
		while (getline(file, line))
		{
			cout << line << '\n';
			int whitespace = line.find(' ');
			int length = line.length();
			string imageFilename = line.substr(0, whitespace);
			string exposureValue = line.substr(whitespace + 1, length);

			// read file
			cout << "Filename: " << imageFilename << endl;
			Mat image = imread(topLevelDirectory + imageFilename);
			images.push_back(image);

			cout << "Exposure: " << exposureValue << endl;
			exposure.push_back(stof(exposureValue));

		}
		file.close();
	}

	Mat HDR = Mat::zeros(images[0].size(), CV_32FC3);

	int maxIter = 30;
	const int rows = images[0].rows;
	const int cols = images[0].cols;
	vector<vector<float>> estimatedX(rows, vector<float>(cols));
	vector<vector<float>> variance(rows, vector<float>(cols));

	// disjoint sets
	vector<int> fa(rows * cols, 0);
	vector<int> memCounter(rows * cols, 0);
	vector<int> discardFactor(rows * cols, 0);

	// create sample image with 3 channels
	int nImages = images.size();
	constexpr int nChannels = 3;
	constexpr int nValues = 256;
	vector<vector<float>> im(nChannels, vector<float>(nValues));

	for (int channel = 0; channel < nChannels; channel++)
	{
		for (int value = 0; value < nValues; value++)
			im[channel][value] = value / 128.f;

		// n iterations 
		for (int i = 0; i < maxIter; i++)
		{
			// iter through every pixel in the image to find X
			for (int r = 0; r < rows; r++)
			{
				for (int c = 0; c < cols; c++)
				{
					estimatedX[r][c] = 0.f;
					float weightedSum = 0.f;
					for (int p = 0; p < nImages; p++)
					{
						int bit8 = images[p].at<Vec3b>(r, c)(channel);

						estimatedX[r][c] += robWeight(bit8) * exposure[p] * im[channel][bit8];
						weightedSum += robWeight(bit8) * exposure[p] * exposure[p];
					}

					estimatedX[r][c] /= weightedSum;
				}
			}
		}

		vector<int> cardinality(256, 0);

		for (int i = 0; i < cardinality.size(); i++)
		{
			im[channel][i] = 0;
		}

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				for (int t = 0; t < nImages; t++)
				{
					int bit8 = images[t].at<Vec3b>(r, c)(channel);
					cardinality[bit8]++;
					im[channel][bit8] += exposure[t] * estimatedX[r][c];
				}
			}
		}

		for (int i = 0; i < cardinality.size(); i++)
		{
			if (cardinality[i] == 0)
				continue;
			im[channel][i] /= cardinality[i];
		}

		float mid = im[channel][128];
		for (int i = 0; i < cardinality.size(); i++)
		{
			im[channel][i] /= mid;
		}
	}

	// variance for irradiance
	for (int channel = 0; channel < 3; channel++)
	{
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				float weightedSum = 0;
				float eSum = 0;  
				float e2Sum = 0;

				for (int t = 0; t < nImages; t++)
				{
					int bit8 = images[t].at<Vec3b>(r, c)(channel);

					float E = im[channel][bit8] / exposure[t];
					eSum += robWeight(bit8) * E;
					e2Sum += robWeight(bit8) * E * E;
					weightedSum += robWeight(bit8);
				}

				float var = (e2Sum / weightedSum) / (eSum * eSum / weightedSum / weightedSum) - 1;
				variance[r][c] = max(variance[r][c], var);
			}
		}
	}


	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			fa[r * cols + c] = r * cols + c;
			memCounter[r * cols + c] = 1;
		}
	}

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			int id = r * cols + c;
			if (variance[r][c] > THRESHOLD)
			{
				for (int dx = -2; dx <= 2; dx++)
				{
					for (int dy = -2; dy <= 2; dy++)
					{
						if (inside(r + dx, c + dy, rows, cols))
							unio(id, id + dx * cols + dy, fa, memCounter);
					}
				}
			}
		}
	}

	vector<int> bestExposure;
	vector<float> bestExposureValue;
	vector<Vec3b> segmentColor;

	int discCounter = 1;

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			int id = r * cols + c;
			if (memCounter[findInSet(id, fa)] > rows * cols * .001)
			{
				if (discardFactor[findInSet(id, fa)] == 0)
				{
					discardFactor[findInSet(id, fa)] = discCounter++;

					bestExposure.push_back(-1);
					bestExposureValue.push_back(0);
					segmentColor.push_back(Vec3b(rand() % 256, rand() % 256, rand() % 256));
				}
				discardFactor[id] = discardFactor[findInSet(id, fa)];
			}
		}
	}
	printf("Moving Segments: %d\n", discCounter - 1);

	// Find the best exposure time for each Disjoint Sets
	for(int t = 0; t < nImages; t++)
	{
		vector<float> expo_val;
		for(int i = 0; i < discCounter - 1; i++)
			expo_val.push_back(0);

		for(int r = 0; r < rows; r++)
		{
			for(int c = 0; c < cols; c++)
			{
				int id = r * cols + c;
				if(discardFactor[findInSet(id, fa)] != 0)
				{
					for(int channel = 0; channel < 3; channel ++)
					{
						int bit8 = images[t].at<Vec3b>(r, c)(channel);
						expo_val[discardFactor[findInSet(id, fa)] - 1] += robWeight(bit8);
					}
				}
			}
		}

		for(int i = 0; i < discCounter - 1; i++)
		{
			if(expo_val[i] > bestExposureValue[i])
			{
				bestExposure[i] = t;
				bestExposureValue[i] = expo_val[i];
			}
		}
	}

	for(int i = 0; i < discCounter - 1; i++)
		printf("%d %f (%d %d %d)\n", bestExposure[i], bestExposureValue[i],
				segmentColor[i][0], segmentColor[i][1], segmentColor[i][2]);

	// Visualize Disjoint Sets of High Var
	Mat varianceSegment = Mat::zeros(images[0].size(), CV_8UC3);
	for(int r = 0; r < rows; r++)
	{
		for(int c = 0; c < cols; c++)
		{
			int id = r * cols + c;
			if(discardFactor[findInSet(id, fa)] != 0)
				varianceSegment.at<Vec3b>(r, c) = segmentColor[discardFactor[findInSet(id, fa)]-1];
		}
	}
	imwrite("var_Rob.jpg", varianceSegment);

	// Response Outputs
	FILE* response_out = fopen("response_Rob.txt", "w");
	for(int channel = 0; channel < 3; channel ++)
	{
		for(int i = 0; i < 256; i++)
			fprintf(response_out, "%f\t", im[channel][i]);
		fprintf(response_out, "\n");
	}
	fclose(response_out);

	// Final HDR
	for (int channel = 0; channel < 3; channel++)
	{
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				float altern = -1;

				if (discardFactor[findInSet(r * cols + c, fa)] != 0)
				{
					int t = bestExposure[discardFactor[findInSet(r * cols + c, fa)] - 1];
					int bit8 = images[t].at<Vec3b>(r, c)(channel);

					altern = im[channel][bit8] / exposure[t];
				}

				float wsum = 0, Isum = 0;

				for (int t = 0; t < nImages; t++)
				{
					int bit8 = images[t].at<Vec3b>(r, c)(channel);

					Isum += robWeight(bit8) * exposure[t] * im[channel][bit8];
					wsum += robWeight(bit8) * exposure[t] * exposure[t];
				}

				// Soft Change should be better
				if (discardFactor[findInSet(r * cols + c, fa)] != 0)
				{
					float alpha = min(variance[r][c] / THRESHOLD, 1.f);
					HDR.at<Vec3f>(r, c)[channel] = alpha * altern + (1 - alpha) * Isum / wsum;
				}
				else
					HDR.at<Vec3f>(r, c)[channel] = Isum / wsum;
			}
		}
	}
	imwrite("hdr_out.jpg", HDR);

	return EXIT_SUCCESS;
}
