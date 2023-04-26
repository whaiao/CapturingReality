// HDR.cpp : Defines the entry point for the application.
//

#include "HDR.h"

using namespace std;
using namespace cv;

float rob_weight(int bit8)
{
	float q = 255 / 4.f;
	float value = bit8 / q - 2.f;
	value = exp(-value * value);
	return value;
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

						estimatedX[r][c] += rob_weight(bit8) * exposure[p] * im[channel][bit8];
						weightedSum += rob_weight(bit8) * exposure[p] * exposure[p];
					}

					estimatedX[r][c] /= weightedSum;
				}
			}
		}
	}
	return 0;
}
