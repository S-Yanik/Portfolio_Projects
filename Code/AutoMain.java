import org.apache.commons.io.FileUtils;
import org.bytedeco.javacpp.hdf5;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;

import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.nd4j.linalg.dataset.DataSet;


public class AutoMain
{


    ///////////////////////////////////////////////////////////////////////////////////////
    //Training Here: /////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //Associated varuables:
    private static int timeBetweenDisplays = 4; //in seconds
    private static int batchSize = 20;

    private static int timeBetweenSaves = 900;//in seconds
    private static boolean loading = false;
    private static boolean saving = false;
    private static String saveName = "humanFaceSave";

    //Associated Variables for dataset
    private static int width = 68;//input data will be scaled down to this width
    private static int height = 68;//by this height
    private static String[] fileTypes = new String[]{ "jpg"}; //what extensions can your files have?
    private static boolean debug = true;
    private static String dataSetLocation = "C:\\Users\\Sean\\Downloads\\face dataset";
    private static int dataSetSize = 2000;
    //Associated Variables
    private static int dimensions = 90;//how many dimensions are we compressing to?
    private static int condensedLayerLocation = 2;// on which layer is there the fewest neurons?
    private static int seed = 735;
    private static int middleLayerNeuronNumber = (60*60);//(int)(2.5*Math.sqrt(width*height));
      private static final IUpdater UPDATER = Adam.builder().learningRate(0.00005).beta1(0.5).build();//AdaGrad.builder().learningRate(0.01).build();//AdaGrad will be updating our network for us, using momentum and the like
    private static boolean changeTheUpdater = false;

    //Associated variables for visualisation:
    private static JFrame frame;
    private static JPanel panel;
    private static int rows = 2; //How many rows of images/graphs/etc will be displayed?
    private static int columns = 2; //how many columns will be displayed?
    private static int scale = 3;


    public static void main(String args[]) throws IOException
    {
        MultiLayerNetwork auto = null;
        //Create the network
        if (!loading)
        {
            System.out.print("Creating the autoencoder...");
             auto = new MultiLayerNetwork(autoEncoderConfig());
        }else{
            System.out.println("Restoring the autoencoder.");
            auto = ModelSerializer.restoreMultiLayerNetwork(System.getProperty("user.home") + "/Downloads/networkSaves/"+ saveName +".zip");
        }


        if (changeTheUpdater)
        {
            //change updater to the updater
            System.out.println("Changing the IUpdater: \n\n");
            MultiLayerNetwork cloneAuto = new MultiLayerNetwork(autoEncoderConfig());
            cloneAuto.init();
            for (int i = 0; i < auto.getLayers().length; i++)
            {
                cloneAuto.getLayer(i).setParams(auto.getLayer(i).params());
            }
            auto = cloneAuto;
        }


        auto.setListeners(new PerformanceListener(timeBetweenDisplays, true));
        System.out.println(" success");

        //Load the data
        System.out.print("Loading our data...");
        File dataLocation = new File(dataSetLocation);//where is the dataset of images
        float[][] dataSetFloats = getData(width,height, dataLocation);

        //the inputs are dataArray, and it is supposed to output dataArray
        System.out.println(" success");

        //set up some variables to record stuff
        int epoch = 0;

        System.out.println("Initiating training!");

        long lastTime = System.nanoTime();
        long timeSinceDisplay = 0;
        long timeSinceSave = 0;
        boolean displayThisEpoch = true;// shall we display the latest generation?
        DataSet thisEpochsDataSet = null;

        while (true)
        {
            epoch++;

            displayThisEpoch = false;


            //calculate time passed
            long thisTime = System.nanoTime();
            long deltaTime = thisTime-lastTime;
            lastTime = thisTime;

            timeSinceDisplay += deltaTime;
            timeSinceSave += deltaTime;

            if (timeSinceDisplay > TimeUnit.SECONDS.toNanos(timeBetweenDisplays))
            {
                timeSinceDisplay = 0;
                displayThisEpoch = true;
            }

            if (timeSinceSave > TimeUnit.SECONDS.toNanos(timeBetweenSaves))
            {
                timeSinceSave = 0;

                if (saving)
                {
                    System.out.println("SAVING...");
                    String downloads = System.getProperty("user.home") + "/Downloads";
                    ModelSerializer.writeModel(auto, new File(downloads + "/networkSaves/" + saveName + ".zip"), true);
                    System.out.println("\n\n\n\n\n\n\n\n\nSAVED!");
                }
            }


            if (displayThisEpoch) System.out.print("Epoch: " + epoch + ". Training!");
            //get data
           float[][] thisEpochsData = new float[batchSize][width*height*3];

            for (int i = 0; i < batchSize; i++)
            {
                thisEpochsData[i] = dataSetFloats[(int)(Math.random()*dataSetFloats.length)];
                INDArray thisEpochsArray = new NDArray(thisEpochsData);
                thisEpochsDataSet = new DataSet(thisEpochsArray,thisEpochsArray);
            }
            auto.fit(thisEpochsDataSet);//training here






            if (displayThisEpoch)
            {
                //Visualize all the stuff
                System.out.println("Visualizing:");
                INDArray[] imagesToDisplay = new INDArray[12];// 6 images

                System.out.print("    Getting Sample Images...");

                //one random image


                    //4 reconstructions
                int locationInDataSet = 0;
                    System.out.println(auto.activateSelectedLayers(0, condensedLayerLocation - 1, thisEpochsDataSet.get(0).getFeatures()));
                    for (int i = 0; i < 12; i++)
                    {
                        if (i%2 == 0)
                        {
                            locationInDataSet =  (int)(Math.random()*dataSetFloats.length);
                            imagesToDisplay[i] = new NDArray(dataSetFloats[locationInDataSet]);
                        }else
                        {
                            imagesToDisplay[i] = auto.activateSelectedLayers(0, auto.getLayers().length - 1,imagesToDisplay[i-1]);
                        }
                    }



                //zeros

                System.out.println(" success");

                display(imagesToDisplay);
            }


        }//infinite learning loop
    }//end of main

    ///////////////////////////////////////////////////////////////////////////////////////
    //The Setup for our network goes here /////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////





    private static MultiLayerConfiguration autoEncoderConfig()
    {
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(UPDATER)//this is a static variable, see above
                 .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .weightInit(WeightInit.XAVIER)//what kind of random does it start with?
                .list()
                .layer(0, new DenseLayer.Builder().nIn(width*height*3).nOut(middleLayerNeuronNumber).activation(Activation.LEAKYRELU).build())
                .layer(1, new DenseLayer.Builder().nIn(middleLayerNeuronNumber).nOut(dimensions).activation(Activation.IDENTITY).build())
                .layer(2, new DenseLayer.Builder().nIn(dimensions).nOut(middleLayerNeuronNumber).activation(Activation.LEAKYRELU).build())//condensedLayerLocation
                .layer(3, new OutputLayer.Builder().nIn(middleLayerNeuronNumber).nOut(width*height*3).activation(Activation.LEAKYRELU)//
                        .lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();


    }





    public static float[][] getData(int width, int height, File whereTheImagesAre) throws IOException
    {


            //File folder = new File(System.getProperty("user.home") + "/Downloads/TrainingFiles/");

            //Gets all of the images as Files
            if (debug) System.out.println("Getting images from " + whereTheImagesAre.getAbsolutePath());
            List<File> listOfFiles =  new ArrayList<>();
            listOfFiles.addAll(FileUtils.listFiles(whereTheImagesAre,fileTypes,true));



            //creates a nice, empty dataset with as many slots as there are files in our folder
            if (debug) System.out.println("Making the empty dataset:");
            float[][]dataSet = new float[(Math.min(dataSetSize,listOfFiles.size()))][width*height*3];
            //float[][]dataSet = new float[listOfFiles.size()][width*height];


            //this loop reads each file and adds it to the dataset.
            //spot keeps track of where in [][]dataSet to add the latest image
            int spot = 0;
            for (int f = 0; f < (Math.min(dataSetSize,listOfFiles.size())); f++)
            {
                File image = listOfFiles.get(f);

                //Display progress
                System.out.print(".");
                if(Math.random() > 0.994)
                {
                    System.out.print(spot);
                }
                if(spot%100 == 0)
                {
                    System.out.println();
                }


                   float[] imageFloats = new float[width * height * 3];//creates an empty array for our latest image
               // float[] imageFloats = new float[width * height];//creates an empty array for our latest image
                    //get and resize the image
                    BufferedImage img = ImageIO.read(image);

                    BufferedImage resized = new BufferedImage(width, height, img.getType());
                    Graphics2D g = resized.createGraphics();
                    g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                            RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                    g.drawImage(img, 0, 0, width, height, 0, 0, img.getWidth(),
                            img.getHeight(), null);
                    g.dispose();

                    img = resized;

                    //the image is now resized


                    //get all the actual values from the image
                    for (int i = 0; i < height * width; i++)
                    {

                        //figure out where from the image to get the color of
                        int x = i % width;
                        int y = i / width;
                        Color atPixel = new Color(img.getRGB(x, y));


                        //get the red green and blue values, add to our array
                        imageFloats[i] = atPixel.getRed() / 255f;
                        imageFloats[i + width * height] = atPixel.getGreen() / 255f;//
                        imageFloats[i + width * height * 2] = atPixel.getBlue() / 255f;

                    }

                    dataSet[spot] = imageFloats;//insert converted image into the dataSet
                    spot++;//increment the spot we are inserting things, so the next image goes in the next spot
            }//end of adding all of the images

            return dataSet;
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    //Methods for displaying stuff here: /////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////


    //Displays what's new with our gan. Takes in a collection of INDArrays to turn into images
    private static void display(INDArray[] images) {
        if (frame == null) //if no frame has been made yet
        {
            //make a new frame
            frame = new JFrame();
            frame.setTitle("Latest Images");
            frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
            frame.setLayout(new BorderLayout());

            panel = new JPanel();

            panel.setLayout(new GridLayout(rows,columns,16,16));
            frame.add(panel, BorderLayout.CENTER);
            frame.setVisible(true);
        }

        panel.removeAll();

        for (int i = 0; i < images.length; i++) {
            //turns the image array into a buffered image which is scaled by a number which is turned into an image icon
            //which is turned into a JLabel, which is then added to the panel.
            panel.add(new JLabel(new ImageIcon(scaleImage(getImageRGB(images[i],width,height),scale,scale))));
        }

        frame.revalidate();
        frame.pack();
    }//end of display

    //gets a RGB buffered image from an INDArray
    //
    //  The inputArray should be structured as follows:
    //  Individual values are the brightness of either red, green or blue
    //  they are calculated by taking, for example, the red value of a pixel, and dividing it by 255 (they are stored as floats, I believe)
    //  This way, if the pixel is all the way red, it will be at the max value of 255, and 255/255 = 1,
    //  if the pixel has no red in it, 0/255 is 0,
    //  and if it is in the middle, say 128, 128/255 == 0.501 ish.
    //
    //The input array must have all of the red values (calculated as above), then all of the greens, then all of the blues
    //
    //The width and height are the width and height of the image
    private static BufferedImage getImageRGB(INDArray inputArray, int width, int height) {
        //the image to return (B uffered I mage --> bi). Sorry, I guess that's pretty obvious.
        BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        //go through the 1/3 of the INDArray (the array is width * height for red, green and blue, so just width
        // times height is 1/3 of it)
        for (int i = 0; i < width*height;i++)//This turns the array into our image
        {
            //calculate where the x and y values should be on the actual image
            int x = i%width;
            int y = i/width;

            //find the reds, greens, and blues
            int red = (int) (255*inputArray.getFloat(i)); //get the red directly from the spot
            int green = (int) (255*inputArray.getFloat(i+width*height)); //get the green from the spot 1/3 of the array forwards
            int blue = (int) (255*inputArray.getFloat(i+width*height*2)); //get the blue from the spot 2/3 of the array forwards

            //There is a possibility that the AI has generated impossible colors
            //(IE darker than black (RGB = -20, -5, 0, for example) or brighter than white
            //clamp these values to get something decent (clamping forces the values to be between the min and max)
            red = clamp(red,0,255);
            green = clamp(green,0,255);
            blue = clamp(blue,0,255);

            //get the color for the pixel (stored as an int)
            int colorOfPixel = new Color(red,green,blue).getRGB();

            //set the color at the spot
            bi.setRGB(x,y,colorOfPixel);

        }//end of coloring in our buffered image. It is now correctly made

        return bi;
    }//end of getImageRGB

    //Scales a buffered image
    private static BufferedImage scaleImage(BufferedImage image, double xScale, double yScale)
    {
        int width = image.getWidth();
        int height = image.getHeight();

        Image imageScaled = null;
        if (xScale > 1&& yScale >1)
        {
            imageScaled = image.getScaledInstance((int) (xScale * width), (int) (yScale * height), Image.SCALE_REPLICATE);
        }
        else
        {
            imageScaled = image.getScaledInstance((int) (xScale * width), (int) (yScale * height), Image.SCALE_REPLICATE);
        }

        BufferedImage bimage = new BufferedImage(imageScaled.getWidth(null), imageScaled.getHeight(null), BufferedImage.TYPE_INT_RGB);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(imageScaled, 0, 0, null);
        bGr.dispose();

        return bimage;
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    //RANDOM UTILITY METHODS HERE:: /////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //clamps value between min and max
    public static int clamp(int value, int min, int max)
    {
        return Math.max(min,Math.min(max,value));
    }


}//The end!