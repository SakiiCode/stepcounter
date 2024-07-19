#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[])
{
  int N;
  float *x, *y, *z;
  int length;
  if (argc > 1)
  {
    N = atoi(argv[1]);
    x = new float[N + 1];
    y = new float[N + 1];
    z = new float[N + 1];
    x[N] = 0;
    y[N] = 0;
    z[N] = 0;
    srand(0);
    for (int i = 0; i < N; i++)
    {
      x[i] = rand() % 10;
      y[i] = rand() % 10;
      z[i] = rand() % 10;
    }
  }
  else
  {
    ifstream inputFile("accelerometer.txt");
    if (inputFile.is_open())
    {
      inputFile >> N;
      x = new float[N + 1];
      y = new float[N + 1];
      z = new float[N + 1];
      x[N] = 0;
      y[N] = 0;
      z[N] = 0;
      for (int i = 0; i < N; i++)
      {
        float elapsedTimeSystem, elapsedTimeSensor, xVal, yVal, zVal;
        inputFile >> elapsedTimeSystem;
        inputFile >> elapsedTimeSensor;
        inputFile >> xVal;
        inputFile >> yVal;
        inputFile >> zVal;
        // cout << xVal << " " << yVal << " " << zVal << endl;
        x[i] = xVal;
        y[i] = yVal;
        z[i] = zVal;
      }
    }
    else
    {
      cerr << "File not open\n";
      return -1;
    }
  }

  auto timeStart = high_resolution_clock::now();
  float *combined = new float[N];
  for (int i = 0; i < N; i++)
  {
    combined[i] = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
    // cout << combined[i] << " ";
  }

  // cout << endl;

  auto lengthEnd = high_resolution_clock::now();

  bool *output = new bool[N];

  for (int elementId = 0; elementId < N; elementId++)
  {
    if (elementId == 0)
    {
      output[elementId] = 0;
    }
    else if (combined[elementId - 1] < combined[elementId] && combined[elementId + 1] < combined[elementId])
    {
      output[elementId] = 1;
    }
    else
    {
      output[elementId] = 0;
    }
  }
  auto peakEnd = high_resolution_clock::now();

  int sum = 0;
  for (int i = 0; i < N; i++)
  {
    sum += output[i];
  }

  auto sumEnd = high_resolution_clock::now();

  /*for (int i = 0; i < N; i++)
  {
    printf("%d %d\n", i, output[i]);
  }
  printf("\n");*/
  cout << "Result: " << sum << endl;

  cout << "Length calculation time:" << duration_cast<microseconds>(lengthEnd - timeStart).count() << " us" << endl;
  cout << "Peak detection time:" << duration_cast<microseconds>(peakEnd - lengthEnd).count() << " us" << endl;
  cout << "Sum time:" << duration_cast<microseconds>(sumEnd - peakEnd).count() << " us" << endl;
  cout << "Total time:" << duration_cast<microseconds>(sumEnd - timeStart).count() << " us" << endl;
  return 0;
}