/*Need to generalize f_theta for Y>1
 *Need to generalize function "train" for Y>1 (specifically where W is updated)
 */
#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>

using namespace std;
extern "C" FILE *popen(const char *command, const char *mode);

#define Train_Set_Size 20
#define PI 3.141592653589793238463
#define N 5
#define epsilon 0.05
#define epoch 50000
#define X 2
#define Y 1
#define MAX 10

double c[N] = {};
double W[N][X] = {}; //(N, X)
double V[N][Y] = {};
double b = 0;

double sigmoid(double x) {
	return (1.0f / (1.0f + exp(-x)));
}

double f_theta(double x[])
{
	double result[Y] = {b};
	double temp=0;
	for (int i = 0; i < N; i++) {
        //FOR SYNAPSE 1
        for(int j=0; j<X; ++j){
            temp += W[i][j]*x[j];
        }
        //FOR SYNAPSE 2
        for(int k=0; k<Y; ++k)
            result[k] += V[i][k] * sigmoid(c[i] + temp);
	}
	return result[0];
}

void train(double x[], double y[])
{
	for (int i = 0; i < N; i++)
    {
        double z = {0};
        for(int j = 0; j<X; ++j)
        {
            z += W[i][j] * x[j];
        }
            z += c[i];  //ADDING THE BIAS
            z = sigmoid(z);

        for(int j = 0; j<X; ++j)
        {
            W[i][j] = W[i][j] - epsilon * (1/Train_Set_Size) *
                    (2 * (f_theta(x) - y[0]) )*   //2(y-y_bar)
                    (V[i][0] *(1 - z) * z)*       //V(1-z)z
                     x[j];
        }
    }

	for (int i = 0; i < N; i++)
	{
        double z = 0;
        for(int j = 0; j<X; ++j)
        {
            z += W[i][j] * x[j];
        }
        z += c[i];
        z = sigmoid(z);

        //auto z = sigmoid(c[i] + W[i] *x);
		V[i][0] = V[i][0] - epsilon * (1/Train_Set_Size)*
                                2 * (f_theta(x) - y[0]) *
                                z;
	}

	b = b - epsilon * (1/Train_Set_Size)* 2 * (f_theta(x) - y[0]);

	for (int i = 0; i < N; i++) {
        double z = 0;
        for(int j = 0; j<X; ++j)
        {
            z += W[i][j] * x[j];
        }
        z += c[i];
        z = sigmoid(z);

		c[i] = c[i] - epsilon * (1/Train_Set_Size) *
                                2 * (f_theta(x) - y[0]) *
                                V[i][0] *
                                (1 - z) * z;
	}
}

int main()
{
	srand(time(NULL));
	//RANDOMLY INITIALISING WEIGHTS
	for (int i = 0; i < N; i++)
        for(int j=0; j<X; ++j)
            W[i][j] = 2 * rand() / RAND_MAX -1;

	for (int i = 0; i < N; i++)
    {
		V[i][0] = 2 * rand() / RAND_MAX -1;
		c[i] = 2 * rand() / RAND_MAX -1;
	}

	//vector<vector<double, double>> input(Train_Set_Size);
    double input [Train_Set_Size][X];
    double output [Train_Set_Size][Y];

    //GENERATING TRAINING DATA
	for (int i = 0; i < Train_Set_Size; i++)
    {
        double temp = 0;
        for(int j=0; j<X; ++j)
            {
                input[i][j] = rand()%MAX;
                temp+= input[i][j];
                input[i][j] /= (MAX*X);
            }
        output[i][1] = temp/(MAX*X);
	}

	for (int j = 0; j < epoch; j++)
    {
		for (int i = 0; i < Train_Set_Size; i++)
		{
			train(input[i], output[i]);
		}
		std::cout << j << "\r";
	}

	//Plot the results
	int Test_Set_Size = 100;
    vector<double> x;
	vector<double> y1, y2;
    double test_data [Test_Set_Size][X];
    for (int i = 0; i < Test_Set_Size; i++)
    {
        double temp = 0;
        for(int j=0; j<X; ++j)
            {
                test_data[i][j] = i%MAX;
                temp+= test_data[i][j];
                test_data[i][j] /= (MAX*X);

            }
        x.push_back(i);
        y1.push_back(temp);
        y2.push_back(20+  (f_theta(test_data[i]) *MAX*X));
        cout<<"actual: "<<temp<<"   pred: "<<(f_theta(test_data[i]) *MAX*X)+20<<endl;
    }

	FILE * gp = popen("gnuplot", "w");
	fprintf(gp, "set terminal wxt size 600,400 \n");
	fprintf(gp, "set grid \n");
	fprintf(gp, "set title '%s' \n", "f(x) = sin (x)");
	fprintf(gp, "set style line 1 lt 3 pt 7 ps 0.1 lc rgb 'blue' lw 1 \n");
	fprintf(gp, "set style line 2 lt 3 pt 7 ps 0.1 lc rgb 'red' lw 1 \n");
	fprintf(gp, "plot '-' w p ls 1, '-' w p ls 2 \n");

	//Exact f(x) = sin(x) -> Green Graph
	for (int k = 0; k < x.size(); k++) {
		fprintf(gp, "%f %f \n", x[k], y1[k]);
	}
	fprintf(gp, "e\n");

	//Neural Network Approximate f(x) = sin(x) -> Red Graph
	for (int k = 0; k < x.size(); k++) {
		fprintf(gp, "%f %f \n", x[k], y2[k]);
	}
	fprintf(gp, "e\n");

	fflush(gp);
	system("pause");
	//pclose(gp);
	return 0;
}
