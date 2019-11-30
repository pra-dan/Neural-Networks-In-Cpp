//Shuffling made right
//Added linear activation beside tanh
#include <iostream>
#include<vector>
#include <math.h>

#define PI 3.141592653589793238463

#define N
#define epsilon 0.05
#define epoch 10000

using namespace std;
extern "C" FILE *popen(const char *command, const char *mode);

///SIGMOID ACTIVATION DEFINITIONS
double sigmoid(double x) { return 1.0f / (1.0f + exp(-x)); }
double dsigmoid(double x) { return x * (1.0f - x); }

///LINEAR ACTIVATION DEFINITIONS
double lin(double x) { return x;}
double dlin(double x) { return 1.0f;}

///TANH ACTIVATION DEFINITIONS
double tanh(double x) { return (exp(x)-exp(-x))/(exp(x)+exp(-x)) ;}
double dtanh(double x) {return 1.0f - x*x ;}

///WEIGHT INITIALIZER
double init_weight() { return (2.*rand()/RAND_MAX -1); }

double MAXX = -9999999999999999; //maximum value of input example
static const int numInputs = 1;
static const int numHiddenNodes = 8;
static const int numOutputs = 1;
static const int numTrainingSets = 100;
const double lr = 0.05f;

double hiddenLayer[numHiddenNodes];
double outputLayer[numOutputs];

double hiddenLayerBias[numHiddenNodes];                     ///BIASES OF HIDDEN LAYER (c)
double outputLayerBias[numOutputs];                         ///BIASES OF OUTPUT LAYER (b)

double hiddenWeights[numInputs][numHiddenNodes];            ///WEIGHTS OF HIDDEN LAYER (W)
double outputWeights[numHiddenNodes][numOutputs];           ///WEIGHTS OF OUTPUT LAYER (V)

double training_inputs[numTrainingSets][numInputs];
double training_outputs[numTrainingSets][numOutputs];

void shuffle(int *array, size_t n)
{
    if (n > 1) //If no. of training examples > 1
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

void predict(double test_sample[])
{
    for (int j=0; j<numHiddenNodes; j++)
    {
        double activation=hiddenLayerBias[j];
        for (int k=0; k<numInputs; k++)
        {
            activation+=test_sample[k]*hiddenWeights[k][j];
        }
        hiddenLayer[j] = tanh(activation);
    }

    for (int j=0; j<numOutputs; j++)
    {
        double activation=outputLayerBias[j];
        for (int k=0; k<numHiddenNodes; k++)
        {
            activation+=hiddenLayer[k]*outputWeights[k][j];
        }
        outputLayer[j] = lin(activation);
    }
    //std::cout<<outputLayer[0]<<"\n";
    //return outputLayer[0];
    //std::cout << "Input:" << training_inputs[i][0] << " " << training_inputs[i][1] << "    Output:" << outputLayer[0] << "    Expected Output: " << training_outputs[i][0] << "\n";
}

int main(int argc, const char * argv[])
{
    ///TRAINING DATA GENERATION
    for (int i = 0; i < numTrainingSets; i++)
    {
		double p = (2*PI*(double)i/numTrainingSets);
		training_inputs[i][0] = p;
		training_outputs[i][0] = 0.2+0.4*pow(p, 2)+0.3*p*sin(15*p)+0.05*cos(50*p);

        /***************************Try Avoiding Edits In This part*******************************/
        ///FINDING NORMALIZING FACTOR
        for(int m=0; m<numInputs; ++m)
            if(MAXX < training_inputs[i][m])
                MAXX = training_inputs[i][m];
        for(int m=0; m<numOutputs; ++m)
            if(MAXX < training_outputs[i][m])
                MAXX = training_outputs[i][m];

        //cout<<"In: "<<training_inputs[i][0]<<"  out: "<<training_outputs[i][0]<<endl;
	}

	///NORMALIZING
	for (int i = 0; i < numTrainingSets; i++)
	{
        for(int m=0; m<numInputs; ++m)
            training_inputs[i][m] /= 1.0f*MAXX;

        for(int m=0; m<numOutputs; ++m)
            training_outputs[i][m] /= 1.0f*MAXX;

        cout<<"In: "<<training_inputs[i][0]<<"  out: "<<training_outputs[i][0]<<endl;
	}
    ///WEIGHT & BIAS INITIALIZATION
    for (int i=0; i<numInputs; i++) {
        for (int j=0; j<numHiddenNodes; j++) {
            hiddenWeights[i][j] = init_weight();
        }
    }
    for (int i=0; i<numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weight();
        for (int j=0; j<numOutputs; j++) {
            outputWeights[i][j] = init_weight();
        }
    }
    for (int i=0; i<numOutputs; i++) {
        //outputLayerBias[i] = init_weight();
        outputLayerBias[i] = 0;
    }

    ///FOR INDEX SHUFFLING
    int trainingSetOrder[numTrainingSets];
    for(int j=0; j<numTrainingSets; ++j)
        trainingSetOrder[j] = j;


    ///TRAINING
    //std::cout<<"start train\n";
    vector<double> performance, epo; ///STORE MSE, EPOCH
    for (int n=0; n < epoch; n++)
    {
        double MSE = 0;
        shuffle(trainingSetOrder,numTrainingSets);
        std::cout<<"\nepoch :"<<n;
        for (int x=0; x<numTrainingSets; x++)
        {
            int i = trainingSetOrder[x];
            //std::cout<<"Training Set :"<<x<<"\n";
            /// Forward pass
            for (int j=0; j<numHiddenNodes; j++)
            {
                double activation=hiddenLayerBias[j];
                //std::cout<<"Training Set :"<<x<<"\n";
                 for (int k=0; k<numInputs; k++) {
                    activation+=training_inputs[i][k]*hiddenWeights[k][j];
                }
                hiddenLayer[j] = tanh(activation);
            }

            for (int j=0; j<numOutputs; j++) {
                double activation=outputLayerBias[j];
                for (int k=0; k<numHiddenNodes; k++)
                {
                    activation+=hiddenLayer[k]*outputWeights[k][j];
                }
                outputLayer[j] = tanh(activation);
            }

            //std::cout << "Input:" << training_inputs[x][0] << " " << "    Output:" << outputLayer[0] << "    Expected Output: " << training_outputs[x][0] << "\n";
            for(int k=0; k<numOutputs; ++k)
                MSE += (1.0f/numOutputs)*pow( training_outputs[i][k] - outputLayer[k], 2);

           /// Backprop
           ///   For V
            double deltaOutput[numOutputs];
            for (int j=0; j<numOutputs; j++) {
                double errorOutput = (training_outputs[i][j]-outputLayer[j]);
                deltaOutput[j] = errorOutput*dtanh(outputLayer[j]);
            }

            ///   For W
            double deltaHidden[numHiddenNodes];
            for (int j=0; j<numHiddenNodes; j++) {
                double errorHidden = 0.0f;
                for(int k=0; k<numOutputs; k++) {
                    errorHidden+=deltaOutput[k]*outputWeights[j][k];
                }
                deltaHidden[j] = errorHidden*dtanh(hiddenLayer[j]);
            }

            ///Updation
            ///   For V and b
            for (int j=0; j<numOutputs; j++) {
                //b
                outputLayerBias[j] += deltaOutput[j]*lr;
                for (int k=0; k<numHiddenNodes; k++)
                {
                    outputWeights[k][j]+= hiddenLayer[k]*deltaOutput[j]*lr;
                }
            }

            ///   For W and c
            for (int j=0; j<numHiddenNodes; j++) {
                //c
                hiddenLayerBias[j] += deltaHidden[j]*lr;
                //W
                for(int k=0; k<numInputs; k++) {
                  hiddenWeights[k][j]+=training_inputs[i][k]*deltaHidden[j]*lr;
                }
            }
        }
        //Averaging the MSE
        MSE /= 1.0f*numTrainingSets;
        //cout<< "  MSE: "<< MSE<<endl;
        ///Steps to PLOT PERFORMANCE PER EPOCH
        performance.push_back(MSE*100);
        epo.push_back(n);
    }

    // Print weights
    std::cout << "Final Hidden Weights\n[ ";
    for (int j=0; j<numHiddenNodes; j++) {
        std::cout << "[ ";
        for(int k=0; k<numInputs; k++) {
            std::cout << hiddenWeights[k][j] << " ";
        }
        std::cout << "] ";
    }
    std::cout << "]\n";

    std::cout << "Final Hidden Biases\n[ ";
    for (int j=0; j<numHiddenNodes; j++) {
        std::cout << hiddenLayerBias[j] << " ";

    }
    std::cout << "]\n";
    std::cout << "Final Output Weights";
    for (int j=0; j<numOutputs; j++) {
        std::cout << "[ ";
        for (int k=0; k<numHiddenNodes; k++) {
            std::cout << outputWeights[k][j] << " ";
        }
        std::cout << "]\n";
    }
    std::cout << "Final Output Biases\n[ ";
    for (int j=0; j<numOutputs; j++) {
        std::cout << outputLayerBias[j] << " ";
    }
    std::cout << "]\n";

    ///Plot the results
	vector<float> x;
	vector<float> y1, y2;
    //double test_input[1000][numInputs];
    int numTestSets = numTrainingSets;
	for (float i = 0; i < numTestSets; i=i+0.25)///Note i
    {
        cout<<i<<endl;
        double p = (2*PI*(double)i/numTestSets);
        x.push_back(p);
		///Actual Result
		double res = 0.2+0.4*pow(p, 2)+0.3*p*sin(15*p)+0.05*cos(50*p);
		y1.push_back(res);
		///To Test
		double temp[1];
		temp[0] = p/MAXX;
		predict(temp);
		y2.push_back(outputLayer[0]*MAXX);
	}

	FILE * gp = popen("gnuplot", "w");
	fprintf(gp, "set terminal wxt size 600,400 \n");
	fprintf(gp, "set grid \n");
	fprintf(gp, "set title '%s' \n", "f(x) = Wiggly Function");
	fprintf(gp, "set style line 1 lt 3 pt 7 ps 0.1 lc rgb 'green' lw 1 \n");
	fprintf(gp, "set style line 2 lt 3 pt 7 ps 0.1 lc rgb 'red' lw 1 \n");
	fprintf(gp, "plot '-' w p ls 1, '-' w p ls 2 \n");


	///Exact f(x) = addition -> Green Graph
	for (int k = 0; k < x.size(); k++) {
		fprintf(gp, "%f %f \n", x[k], y1[k]);
	}
	fprintf(gp, "e\n");

	///Neural Network Approximate -> Red Graph
	for (int k = 0; k < x.size(); k++) {
		fprintf(gp, "%f %f \n", x[k], y2[k]);
	}
	fprintf(gp, "e\n");

	fflush(gp);

	///FILE POINTER FOR SECOND PLOT (PERFORMANCE GRAPH)
    FILE * gp1 = popen("gnuplot", "w");
	fprintf(gp1, "set terminal wxt size 600,400 \n");
	fprintf(gp1, "set grid \n");
	fprintf(gp1, "set title '%s' \n", "Performance");
	fprintf(gp1, "set style line 1 lt 3 pt 7 ps 0.1 lc rgb 'green' lw 1 \n");
	fprintf(gp1, "set style line 2 lt 3 pt 7 ps 0.1 lc rgb 'red' lw 1 \n");
	fprintf(gp1, "plot '-' w p ls 1 \n");

    for (int k = 0; k < epo.size(); k++) {
		fprintf(gp1, "%f %f \n", epo[k], performance[k]);
	}
	fprintf(gp1, "e\n");

	fflush(gp1);

	system("pause");
	//pclose(gp);

    return 0;
}
