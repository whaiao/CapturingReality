#include "Tonemap.h"

using namespace std;
using namespace cv;

float transform(float r)
{
	if (r <= .05f)
		return r * 2.64f;

	return 1.099f * pow(r, 0.9f / 2.2f) - 0.099f;
}

void tonemap(Mat hdrMat, Mat tonemapMat)
{
	// source: https://en.wikipedia.org/wiki/SRGB#From_sRGB_to_CIE_XYZ
	int rows = hdrMat.rows;
	int cols = hdrMat.cols;

	struct Pixel {
		float r, g, b;
	};

	vector<vector<Pixel>> pixels(rows, vector<Pixel>(cols));

	float r, g, b;
	float maximumLuminance = -1.f; 
	float base = .75f;
	for (int i = 0; i < rows; i++) 
	{
		for (int j = 0; j < cols; j++) 
		{
			// linearize rgb values
			b = hdrMat.at<Vec3f>(i, j)(0) / 255.0f;
			g = hdrMat.at<Vec3f>(i, j)(1) / 255.0f;
			r = hdrMat.at<Vec3f>(i, j)(2) / 255.0f;

			// CIE XYZ
			pixels[i][j].r = (0.4124f*r + 0.3576f*g + 0.1805f*b);
			pixels[i][j].g = (0.2126f*r + 0.7152f*g + 0.0722f*b);
			pixels[i][j].b = (0.0193f*r + 0.1192f*g + 0.9505f*b);
			maximumLuminance = max(maximumLuminance, pixels[i][j].g);
		}
	}

	for (int i = 0; i < rows; i++) 
	{
		for (int j = 0; j < cols; j++) 
		{
			// normalize luminosity
			float xx = pixels[i][j].r / (pixels[i][j].r + pixels[i][j].g + pixels[i][j].b);
			float yy = pixels[i][j].g / (pixels[i][j].r + pixels[i][j].g + pixels[i][j].b);
			float tp = pixels[i][j].g;

			// eq 4 paper
			pixels[i][j].g = 1.f * log(pixels[i][j].g + 1) / log(2 + 8.0*pow((pixels[i][j].g / maximumLuminance), log(base) / log(0.5))) / log10(maximumLuminance + 1);
			float r = pixels[i][j].g / yy*xx;
			float g = pixels[i][j].g;
			float b = pixels[i][j].g / yy*(1 - xx - yy);

			// recover sRGB values
			r = 3.2410f*r - 1.5374f*g - 0.4986f*b;
			g = -0.9692f*r + 1.8760f*g + 0.0416f*b;
			b = 0.0556f*r - 0.2040f*g + 1.0570f*b;

			vector<float> rgb = { r, g, b };

			// clip values
			for (int channel = 0; channel < rgb.size(); channel++) 
			{
				if (rgb[channel] < 0)
					rgb[channel] = 0;
				if (rgb[channel] > 1)
					rgb[channel] = 1;

				// transform values with inverse power curve
				rgb[channel] = transform(rgb[channel]);

				// copy to output
				tonemapMat.at<Vec3b>(i, j)[0] = int(rgb[2] * 255);
				tonemapMat.at<Vec3b>(i, j)[1] = int(rgb[1] * 255);
				tonemapMat.at<Vec3b>(i, j)[2] = int(rgb[0] * 255);
			}
		}
	}
}
