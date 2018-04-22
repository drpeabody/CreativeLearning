package creativelearning;

/**
 *
 * @author Abhishek
 */
public class CreativeLearning {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        CreativeNetwork c = new CreativeNetwork(1, 1, new int[]{2, 3});
        
        float learn = 0.00001f;
        
        for (float i = 0; i < 200000; i += 1f) {
            TrainingData k = new TrainingData(new float[]{i}, new float[]{(float)(Math.sin(i))});
            c.SGD_train_single(k, learn);
        }
        System.out.println("Training Complete.");
        
//        c.debug();
        
        float err = 0f;
        
        for (float i = 0.1f; i < 10.1f; i+=0.1f) {
            c.setInput(new float[]{i});
            float out[] = c.evaluate();
            err += (float)Math.sin(i) - out[0];
        }
        
        System.out.println("Average Error: " + (err) + "%");
    }
    
}
