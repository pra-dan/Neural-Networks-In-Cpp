/*Y = R+B+G
 *Author: Prashant Dandriyal
 *Date: 14 Dec, 2019
 *-Added BUILTIN_LED glow feature for post-training
 *-Corrected activation of output layer from tanh to lin
 *-Last: 0.05 @250 epochs,6 Hidden Neurons 
 *
 *-Edit (24-01-2020)
 *-Added Elu+Linear activation and corrected some parts of normalization. Got high accuracy.
 *
 *Edit (03-02-2020)
 *-Added momentum and automatic loop break using threshold_error. Also, used a timer to measure execution time.
 */
#define PI 3.141592653589793238463

#define N
#define epsilon 0.05
#define alpha 1.0f
#define epoch 8000
#define momentum 0.09
#define error_threshold 1.e-005

///ELU ACTIVATION DEFINITIONS
float elu(float x) { if(x>0)  return x;
                       else return alpha*(exp(x)-1.0f);
                     }
float delu(float x) { if(x>0) return 1.0f;
                       else return alpha*exp(x);
                      }
///LINEAR ACTIVATION DEFINITIONS
float lin(float x) { return x;}
float dlin(float x) { return 1.0f;}

///tanh ACTIVATION DEFINITIONS
float Tanh_(float x) { return (exp(x)-exp(-x))/(exp(x)+exp(-x)) ;}
float dTanh_(float x) {return 1.0f - x*x ;}

///WEIGHT INITIALIZER
float init_weight() { Serial.println(random(1,10));
                      int temp =  rand()%10;
                      if(temp <6) return -1;
                      else return 0;}

float MAXX = -9999999999999999; //maximum value of input example
static const int numInputs = 3;
static const int numHiddenNodes = 9;
static const int numOutputs = 1;
static const int numTrainingSets = 8;
static const int numTestSets = 16;
const float lr = 0.05f;

float hiddenLayer[numHiddenNodes];
float outputLayer[numOutputs];

float hiddenLayerBias[numHiddenNodes];                     ///BIASES OF HIDDEN LAYER (c)
float outputLayerBias[numOutputs];                         ///BIASES OF OUTPUT LAYER (b)

float hiddenWeights[numInputs][numHiddenNodes];            ///WEIGHTS OF HIDDEN LAYER (W)
float outputWeights[numHiddenNodes][numOutputs];           ///WEIGHTS OF OUTPUT LAYER (V)

float del_hiddenWeights[numInputs][numHiddenNodes] = {0};
float del_outputWeights[numHiddenNodes][numOutputs] = {0};

float training_inputs[numTrainingSets][numInputs] = { {255,255,255},
                                                       {255,218,185},
                                                       {245,255,250},
                                                       {230,230,250},
                                                       {0,255,0},
                                                       {0,100,0},
                                                       {46,139,87},
                                                       {127,255,0}
                                                     };

 float test_inputs[numTestSets][numInputs] =        { {0,0,255},
                                                       {135,206,235},
                                                       {175,238,238},
                                                       {127,255,212},
                                                       {255,0,0},
                                                       {255,69,0},
                                                       {255,127,0},
                                                       {255,165,0},
                                                       {0,0,0},
                                                       {105,105,105},
                                                       {112,112,112},
                                                       {169,169,169},
                                                       {155,48,255},
                                                       {139,35,35},
                                                       {205,51,51},
                                                       {255,246,143},
                                                     };

float training_outputs[numTrainingSets][numOutputs] ={ {765},
                                                        {658},
                                                        {750},
                                                        {710},
                                                        {255},
                                                        {100},
                                                        {272},
                                                        {382}
                                                      };

float test_outputs[numTestSets][numOutputs] =       {  {255},
                                                        {576},
                                                        {651},
                                                        {594},
                                                        {255},
                                                        {324},
                                                        {382},
                                                        {420},
                                                        {0},
                                                        {315},
                                                        {336},
                                                        {507},
                                                        {458},
                                                        {209},
                                                        {307},
                                                        {664}
                                                       };

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

void predict(float test_sample[])
{
    for (int j=0; j<numHiddenNodes; j++)
    {
        float activation=hiddenLayerBias[j];
        for (int k=0; k<numInputs; k++)
        {
            activation+=test_sample[k]*hiddenWeights[k][j];
        }
        hiddenLayer[j] = elu(activation);
    }

    for (int j=0; j<numOutputs; j++)
    {
        float activation=outputLayerBias[j];
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

void setup() 
{  
  Serial.begin(9600);
  //TO TIME THE CODE
  unsigned long time = millis();
  
  randomSeed(analogRead(0));
  pinMode(13,OUTPUT);
  
  ///TRAINING DATA GENERATION
    for (int i = 0; i < numTrainingSets; i++)
    {
    /*
    float p = (2*PI*(float)i/numTrainingSets);
    training_inputs[i][0] = p;
    training_outputs[i][0] = (0.2+0.4*pow(p, 2)+0.3*p*sin(15*p)+0.05*cos(50*p))/100.;
    */  
        ///APPLYING CHROMATICITY CALCULATION
        training_inputs[i][0] = training_inputs[i][0]*(-0.14282) + training_inputs[i][1]*(1.54924) + training_inputs[i][2]*(-0.95641);
        training_inputs[i][1] = training_inputs[i][0]*(-0.32466) + training_inputs[i][1]*(1.57837) + training_inputs[i][2]*(-0.73191);
        training_inputs[i][2] = training_inputs[i][0]*(-0.68202) + training_inputs[i][1]*(0.77073) + training_inputs[i][2]*(0.56332);

        training_outputs[i][0] = training_inputs[i][0]+training_inputs[i][1]+training_inputs[i][2];

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
            training_inputs[i][m] /= 1.0f*MAXX*numInputs;

        for(int m=0; m<numOutputs; ++m)
            training_outputs[i][m] /= 1.0f*MAXX*numInputs;

        Serial.print("In: ");
        Serial.print(training_inputs[i][0], 3);
        Serial.print("  out: ");
        Serial.println(training_outputs[i][0], 3);
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
    float MSE = 0;
    int n=0;
    for (n=0; n < epoch; n++)
    {
        shuffle(trainingSetOrder,numTrainingSets);
        Serial.print("\nepoch : ");Serial.print(n);
        for (int x=0; x<numTrainingSets; x++)
        {
            int i = trainingSetOrder[x];
            //std::cout<<"Training Set :"<<x<<"\n";
            /// Forward pass
            for (int j=0; j<numHiddenNodes; j++)
            {
                float activation=hiddenLayerBias[j];
                //std::cout<<"Training Set :"<<x<<"\n";
                 for (int k=0; k<numInputs; k++) {
                    activation+=training_inputs[i][k]*hiddenWeights[k][j];
                }
                hiddenLayer[j] = elu(activation);
            }

            for (int j=0; j<numOutputs; j++) {
                float activation=outputLayerBias[j];
                for (int k=0; k<numHiddenNodes; k++)
                {
                    activation+=hiddenLayer[k]*outputWeights[k][j];
                }
                outputLayer[j] = lin(activation);
            }

            //std::cout << "Input:" << training_inputs[x][0] << " " << "    Output:" << outputLayer[0] << "    Expected Output: " << training_outputs[x][0] << "\n";
            for(int k=0; k<numOutputs; ++k)
                MSE += (1.0f/numOutputs)*pow( training_outputs[i][k] - outputLayer[k], 2);

           /// Backprop
           ///   For V
            float deltaOutput[numOutputs];
            for (int j=0; j<numOutputs; j++) {
                float errorOutput = (training_outputs[i][j]-outputLayer[j]);
                deltaOutput[j] = errorOutput*dlin(outputLayer[j]);
            }

            ///   For W
            float deltaHidden[numHiddenNodes];
            for (int j=0; j<numHiddenNodes; j++) {
                float errorHidden = 0.0f;
                for(int k=0; k<numOutputs; k++) {
                    errorHidden+=deltaOutput[k]*outputWeights[j][k];
                }
                deltaHidden[j] = errorHidden*delu(hiddenLayer[j]);
            }

            ///Updation
            ///   For V and b
            for (int j=0; j<numOutputs; j++) {
                //b
                outputLayerBias[j] += deltaOutput[j]*lr;
                for (int k=0; k<numHiddenNodes; k++)
                {
                    del_hiddenWeights[k][j] = (hiddenLayer[k]*deltaOutput[j]*lr) + momentum*del_hiddenWeights[k][j];
                    outputWeights[k][j]+= hiddenLayer[k]*deltaOutput[j]*lr;
                }
            }

            ///   For W and c
            for (int j=0; j<numHiddenNodes; j++) {
                //c
                hiddenLayerBias[j] += deltaHidden[j]*lr;
                //W
                for(int k=0; k<numInputs; k++) {
                  del_hiddenWeights[k][j] = (training_inputs[i][k]*deltaHidden[j]*lr) + momentum*del_hiddenWeights[k][j];
                  hiddenWeights[k][j]+=training_inputs[i][k]*deltaHidden[j]*lr;
                }
            }
        }
        //Averaging the MSE
        MSE /= 1.0f*numTrainingSets;
        Serial.print("\t          Error: ");Serial.println(MSE,8);
        if(MSE < error_threshold)
            break;
    }
    Serial.print("TRAINING TIME: ");Serial.print(millis()-time); Serial.println(" ms");
    
    Serial.print("Final MSE: ");Serial.print(MSE,10);
    Serial.print(" at: ");Serial.println(n-1);
    //TRAINING COMPLETE
    digitalWrite(13,HIGH);
    
    Serial.print("Final Hidden Weights\n");
    for (int j=0; j<numHiddenNodes; j++) {
        Serial.print("[ ");
        for(int k=0; k<numInputs; k++) {
            Serial.print(hiddenWeights[k][j], 9);
        }
        Serial.print("] \n");
    }

    //Predict
    //int numTestSets = numTrainingSets;
    for (int i = 0; i < numTestSets; i++)///Note i
    {
      //float p = (2*PI*(float)i/numTestSets);
      ///APPLYING CHROMATICITY CALCULATION
      test_inputs[i][0] = test_inputs[i][0]*(-0.14282) + test_inputs[i][1]*(1.54924) + test_inputs[i][2]*(-0.95641);
      test_inputs[i][1] = test_inputs[i][0]*(-0.32466) + test_inputs[i][1]*(1.57837) + test_inputs[i][2]*(-0.73191);
      test_inputs[i][2] = test_inputs[i][0]*(-0.68202) + test_inputs[i][1]*(0.77073) + test_inputs[i][2]*(0.56332);

      test_outputs[i][0] = test_inputs[i][0]+test_inputs[i][1]+test_inputs[i][2];

      ///Actual Result
      Serial.print("Expected: ");
      Serial.print(test_outputs[i][0], 3);

      test_inputs[i][0] /= MAXX*numInputs;
      test_inputs[i][1] /= MAXX*numInputs;
      test_inputs[i][2] /= MAXX*numInputs;

      //Make Prediction
      predict(test_inputs[i]);
      
      float res = MAXX*numInputs*outputLayer[0];
      Serial.print("\tPredicted: ");Serial.println(res,3);
      // Serial.println(2.*rand()/RAND_MAX -1);
    }
    
}

void loop() 
{
  // put your main code here, to run repeatedly:

}
