package creativelearning;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * @since Apr 21, 2018
 * @author Abhishek
 */
public class CreativeNetwork {
    static interface Mapper{
        public float wrap(Node l);
    }
    static interface NodeMap{
        public Node map(Node n);
    }
    
    static class Layer{
        Node[] nodes;
        
        Layer(int numNodes, NodeMap r){
            this.nodes = new Node[numNodes];
            for (int i = 0; i < numNodes; i++) {
                nodes[i] = r.map(new Node());
            }
        }
    }
    static class Node{
        Layer pre;
        float w[];//both are arrays of same length
        float bias, err, out;
        
        Node init(Layer previous){
            pre = previous;
            w = new float[pre.nodes.length];
            for (int i = 0; i < w.length; i++) {
                w[i] = (float)Math.random();
            }
            bias = (float)Math.random();
            err = 0f;
            out = 0f;
            return this;
        }
        
        public float getRes(){
            float f = 0.0f;
            for(int i = 0; i < w.length; i++){
                f += w[i] * pre.nodes[i].getRes();
            }
            out = f;
            return (float)(1/(1+Math.pow(Math.E, -f - bias)));
        }
    }
    
    static class InputNode extends Node{
        public float f;
        @Override
        public float getRes(){
            return f;
        }
    }
    
    final Layer input, output, hidden[];
    
    CreativeNetwork(int inputs, int outputs, int nodesPerHiddenLayer[]) {
        input = new Layer(inputs, (Node n) -> new InputNode());
        hidden = new Layer[nodesPerHiddenLayer.length];
        
        for (int i = 0; i < nodesPerHiddenLayer.length; i++) {
            Layer preLayer = (i == 0)? input: hidden[i - 1];
            hidden[i] = new Layer(nodesPerHiddenLayer[i], (Node n) -> n.init(preLayer));
        }
        output = new Layer(outputs, (Node n) -> n.init(hidden[hidden.length - 1]));
    }
    
    public void setInput(float f[]){
        if(f.length != input.nodes.length) throw new IllegalArgumentException();
        for(int i = 0; i < f.length; i++){
            ((InputNode)input.nodes[i]).f = f[i];
        }
    }
    
    void SGD_train_single(TrainingData t, float learning) {
        setInput(t.inputs);
        calculateOutPutLayerError(t.outputs);
        for (int i = hidden.length - 1; i >= 0; i--) {
            for (int j = 0; j < hidden[i].nodes.length; j++) {
                float out = hidden[i].nodes[j].getRes();
                float err = 0f;

                Layer nextLayer = (i == hidden.length - 1) ? output : hidden[i + 1];

                for (Node node : nextLayer.nodes) {
                    err += node.w[j] * node.err;
                }
                err = err * out * (1 - out);
                hidden[i].nodes[j].err = err;
            }
        }
        for (int i = hidden.length; i > 0; i--) {
            Layer preLayer = hidden[i - 1], currLayer = (i == hidden.length)? output: hidden[i];
            for (int j = 0; j < preLayer.nodes.length; j++) {
                for (Node node : currLayer.nodes) {
                    node.w[j] = node.w[j] + learning * node.err * node.out;
                    node.bias = node.bias + learning * node.err;
                }
            }
        }
    }
    void SGD_train(TrainingData input[], int epochs, int batch_size, float learningRate){
        System.out.println("Beginning Training...");
        List l = Arrays.asList(input);
        
        for (int i = 0; i < epochs; i++) {
            Collections.shuffle(l);
            for (int j = 0; j < input.length; j += batch_size) {
                for (int k = j; k < j + batch_size; k++) {
                    SGD_train_single(input[k], learningRate);
                }
            }
            System.out.println("Epochs Completed: " + (i+1) + " out of " + epochs);
        }
    }
    
    public void printInputs(){
        float f[] = wrapLayer(input, (Node l) -> l.getRes());
        System.out.println("Current Input: " + Arrays.toString(f));
    }
    
    float[] wrapLayer(Layer l, Mapper w){
        float f[] = new float[l.nodes.length];

        for (int i = 0; i < f.length; i++) {
            f[i] = w.wrap(l.nodes[i]);
        }
        return f;
    }
    
    public void debug(){
        float f[];
        int x = 0;
        
        printInputs();
        
        for(Layer l: hidden){
            f = wrapLayer(l, (Node n) -> n.getRes());
            System.out.println("Output of Layer " + x++ + " is: " + Arrays.toString(f));
        }
        
        System.out.println("Final Output: " + Arrays.toString(evaluate()));
    }
    public float[] evaluate(){
        float[] outputs = new float[output.nodes.length];
        
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = output.nodes[i].getRes();
        }
        
        return outputs;
    }
    public void calculateOutPutLayerError(float outputs[]){
        for (int i = 0; i < outputs.length; i++) {
            output.nodes[i].err = outputs[i] - output.nodes[i].getRes();
        }
    }
    
}


