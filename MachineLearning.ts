/*/
© BombaskiGeneralZPodlasia // https://github.com/BombaskiGeneralZPodlasia 
/*/

const small = -10
const big = 10
         
function IfNil(Neuron:NeuronClass):number
{
   return 0;
}     
      
function Add(Connection:Connection)
{
 Connection.NeuronB.Value += Connection.NeuronA.ActivationFunction(Connection.NeuronA.Value+Connection.NeuronA.Bias)*Connection.Weight
}     
      
class NonAndLinearFunctions
{
 constructor(){}
 Sigmoid(X:number):number 
 {
    return (1/(1+Math.pow(Math.E,-X)));
 }
 Tanh(X:number):number
 {
    return (this.Sigmoid(X*2)-0.5)*4;
 }
 ReLU(X:number):number
 {
    return Math.max(0,X);
 }
 LeakyReLU(X:number):number
 {
    return Math.max(X*0.01,X);
 }
 SWISCH(X:number):number
 {
    return ((X)/(1+Math.pow(Math.E,-(X))));
 }
 Linear(X:number):number
 {
   return X;
 }
}        
class NeuronClass
{
 Connections:Connection[] = new Array[0];
 IsConnectedBy:Connection[] = new Array[0];
 MaxSubiterations:number = 0;
 Id:number = 0;
 CustomStartFunction:Function;
 ActivationFunction:Function;
 Value:number = 0;
 errorMarginValue:number = 0; 
 BiasChange:number = 0;
 Bias:number = 0;
 Iterations:number = 0;
 BackIterations:number = 0;
 SubIterations:number = 0;
 constructor (CustomStartFunction:Function,ActivationFunction:Function,startingBias:number,MaxSubiterations:number)
 {
  this.CustomStartFunction = CustomStartFunction || IfNil; //it's only parameter is the Neuron and it's expected to return a number
  this.Value = 0;
  this.Bias = startingBias;
  this.ActivationFunction = ActivationFunction;
  this.MaxSubiterations = MaxSubiterations
 }
}
class Connection
{
 NeuronA:NeuronClass;
 NeuronB:NeuronClass;
 Weight:number;
 Id:number = 0;
 NextIncrease:number = 0;//this value is stored for updating connection weights after the training
 constructor(NeuronA:NeuronClass,NeuronB:NeuronClass,Weight:number)
 {
  this.NeuronA = NeuronA
  this.NeuronB = NeuronB
  this.Weight = Weight
  this.NextIncrease = 0;
 }
}     
class NeuralNetwork {
   InputNeurons:NeuronClass[];
   OutputNeurons:NeuronClass[];
   Neurons:NeuronClass[];
   ConnectionCount:number;
   NeuronCount:number; /*/fucking useless semicolons ASI will do the job for me/*/
   CanInvokeCustomFunction:boolean;
   NeuralNetworkSettings:NeuralNetworkOptions;
   NeuralNetworkIdentification:string;
   constructor (InputSize:number,OutputSize:number,NeuralNetworkSettings:NeuralNetworkOptions,CanInvokeCustomFunction:boolean)
   {
    this.InputNeurons = new Array(InputSize)
    this.OutputNeurons = new Array(OutputSize)
    this.Neurons = new Array()
    this.NeuronCount = 0
    this.NeuralNetworkSettings = NeuralNetworkSettings
    this.CanInvokeCustomFunction = CanInvokeCustomFunction;
   }  
   ForwardPropagate ():void
   {
    //This function is semi-recurisve
    let Checked = new Array()
    let NextNeurons = new Array()
    let CurrentNeurons = this.InputNeurons

    this.NeuralNetworkSettings.Iterations += 1

    while (CurrentNeurons.length > 0)
    {
     NextNeurons = new Array()
     Checked = new Array()
     for (let i = 0;i < CurrentNeurons.length;i++) {
      let Neuron:NeuronClass = CurrentNeurons[i]
      if (Neuron.Iterations < this.NeuralNetworkSettings.Iterations)
      {
       Neuron.Iterations += 1
       Neuron.SubIterations = 0
       Neuron.Value = this.CanInvokeCustomFunction ? Neuron.CustomStartFunction(this) : Neuron.Value
      }
      if (Neuron.SubIterations == Neuron.MaxSubiterations)
      {
       continue
      }
      Neuron.SubIterations += 1
      for (let i = 0;i<Neuron.Connections.length;i++)
      {
       Add(Neuron.Connections[i])
       /*/Don't add it do the array if the neuron exists already/*/
       if (Checked[Neuron.Connections[i].NeuronB.Id])
       {
         continue
       }
       Checked[Neuron.Connections[i].NeuronB.Id] = true
       NextNeurons[NextNeurons.length] = Neuron.Connections[i].NeuronB
       Checked[Neuron.Id] = true
      }
     }
     CurrentNeurons = NextNeurons
    }
   }  
   BackwardPropagate(TrainingExamples:TrainingExample[]):void
   {  
    /*/ The computational complexity of that function O_O AND WHY CPU!!!!! USE GPU YOU DUMBASS!!!!/*/
    /*/Determine change/*/
    this.CanInvokeCustomFunction = false
    for (let i = 0;i < TrainingExamples.length;i++)
    { 
      let TrainingExample = TrainingExamples[i]
      /*/Forward Propagation/*/
      for (let i = 0;i < TrainingExample.Inputs.length;i++)
      {
       this.InputNeurons[i].Value = TrainingExample.Inputs[i]
      }
      this.ForwardPropagate()
      /*/ Correction \ Backpropagation /*/
      /*/ Cost calculation is useless here /*/
      /*/ 
       Iterate over every neuron, beginning at output layer and update variables accordingly.
      /*/
      let CurrentNeurons = this.OutputNeurons
      let NextNeurons = new Array()
      let Checked = new Array()
      for (let i = 0;i < CurrentNeurons.length;i++)
      {
       let Neuron:NeuronClass = CurrentNeurons[i]
       Neuron.errorMarginValue = Neuron === TrainingExample.ExpectedOutput ? 1 - this.CalculateConfidence(Neuron) : this.CalculateConfidence(Neuron) - 1
       Neuron.SubIterations = 1
       Neuron.BackIterations += 1
       for (let i = 0;i < Neuron.IsConnectedBy.length;i++)
       {
         let ConnectedNeuron:NeuronClass =  Neuron.IsConnectedBy[i].NeuronA
         NextNeurons[NextNeurons.length] = ConnectedNeuron
       }
      }
      CurrentNeurons = NextNeurons
      this.NeuralNetworkSettings.BackIterations += 1
      while (CurrentNeurons.length > 0)
      { 
       NextNeurons = new Array()
       for (let i = 0;i < CurrentNeurons.length;i++)
       {
         let Neuron:NeuronClass = CurrentNeurons[i]
         if (Neuron.BackIterations < this.NeuralNetworkSettings.BackIterations)
         {
          Neuron.BackIterations += 1 
          Neuron.SubIterations = 0
          Neuron.errorMarginValue = 0
          for (let i = 0;i < Neuron.Connections.length;i++)
          {
            Neuron.errorMarginValue += Neuron.Connections[i].NeuronB.errorMarginValue *  Neuron.Connections[i].Weight
          }
         }
         if (Neuron.SubIterations == Neuron.MaxSubiterations)
         {
          continue
         }
         Neuron.SubIterations += 1
         for (let i = 0;i < Neuron.Connections.length;i++)
         {
           Neuron.Connections[i].Weight += Neuron.Connections[i].Weight * Neuron.Connections[i].NeuronB.errorMarginValue * this.NeuralNetworkSettings.LearnFactor / TrainingExamples.length
           Neuron.Bias += Neuron.Connections[i].NeuronB.errorMarginValue * this.NeuralNetworkSettings.LearnFactor / TrainingExamples.length
         }
       }
      }
      
    } 
    /*/Update network/*/
    let CurrentNeurons = this.InputNeurons
    let Checked = new Array()

    while (CurrentNeurons.length > 0)
    { 
     let NextNeurons = new Array()
     for (let i = 0;i < CurrentNeurons.length;i++)
     {
      let neuron:NeuronClass = CurrentNeurons[i]
      for (let i = 0;i < neuron.Connections.length;i++)
      {
         /*/Specific connections can be iterated a few times only/*/
         if (Checked[neuron.Connections[i].Id])
         {
          continue  
         }
         Checked[neuron.Connections[i].Id] = true
         NextNeurons[NextNeurons.length] = neuron.Connections[i].NeuronB
         neuron.Connections[i].Weight += neuron.Connections[i].NextIncrease
      }
     }
     CurrentNeurons = NextNeurons
    }
    /*/Rest/*/
    this.CanInvokeCustomFunction = true
   }
   CalculateConfidence(Neuron:NeuronClass):number
   {
     let limu = Neuron.ActivationFunction(big)
     let limd = Neuron.ActivationFunction(small)
     let value = Neuron.ActivationFunction(Neuron.Value)+limu
     let dist = Math.abs(limd-limu)
     let confidence = value/dist //confidence
     return confidence
   }
   CreateConnection(NeuronA:NeuronClass,NeuronB:NeuronClass,Weight:number):Connection
   {
    let newConnection:Connection = new Connection(NeuronA,NeuronB,Weight) 
    NeuronA.Connections[NeuronA.Connections.length-1] = newConnection
    NeuronB.IsConnectedBy[NeuronB.IsConnectedBy.length-1] = newConnection
    this.ConnectionCount += 1
    newConnection.Id = this.ConnectionCount
    return newConnection
   }
   AddNeuron(NeuronType:string,CustomStartFunction:Function,ActivationFunction:Function,startingBias:number,MaxSubiterations:number):NeuronClass
   {
    let Neuron:NeuronClass = new NeuronClass(CustomStartFunction,ActivationFunction,startingBias,MaxSubiterations || 1)
    let Tab = new Array()
    if (NeuronType=="Input")
    {
     Tab = this.InputNeurons
    }
    else if(NeuronType=="Output")
    {
     Tab = this.OutputNeurons
    }
    Neuron.Id = this.NeuronCount
    this.Neurons[this.Neurons.length] = Neuron
    this.NeuronCount += 1
    if (!Tab) 
    {
      return Neuron
    }
    Tab[Tab.length] = Neuron
    return Neuron
   }
}
class NeuralNetworkOptions {
 Iterations:number = 0;
 BackIterations:number = 0;
 LearnFactor:number = 0.1;
 constructor ()
 {}
}
class TrainingExample {
   ExpectedOutput:NeuronClass;
   NeuralNetwork:NeuralNetwork;
   Inputs:number[];
   constructor (Inputs:number[],NeuralNetworkT:NeuralNetwork,ExpectedOutput:NeuronClass)
   {
    this.NeuralNetwork = NeuralNetworkT
    this.ExpectedOutput = ExpectedOutput
    this.Inputs = Inputs
   }
}