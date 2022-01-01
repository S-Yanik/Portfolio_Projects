import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.nativeblas.Nd4jCpu;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;


public class   AutoArt
{

    //@Todo
    //Actual AutoEncoding network
    //Displayer
    //Saving, loading, etc

    ///////////////////////////////////////////////////////////////////////////////////////
    //Training Here: /////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //Associated varuables:
    private static int timeBetweenDisplays = 4; //in seconds
    private static int batchSize = 20;

    private static int timeBetweenSaves = 450;//in seconds
    private static boolean loading = true;
    private static boolean saving = true;
    private static String saveName = "autoSaveArt";

    //Associated Variables for dataset
    private static int smallW = 14;
    private static int smallH = 14;
    private static int bigW = 28;//input data will be taken from in snippets of this width
    private static int bigH = 28;//by this height
    private static int paddingAmount = 2;
    private static int secondLayerNumbs = 350;
    private static int thirdLayerNumbs = 650;
    private static int dataSetMultiplication = 45;
    private static int triesPerImage = 5;


    private static String dataSetImageLocation = "C:\\Users\\Sean\\Downloads\\ExportImages";
    private static String[] fileTypes = new String[]{ "png"}; //what extensions can your files have?
    private static boolean debug = true;
    //Associated Variables
    private static int seed = 735;
    private static final IUpdater UPDATER = Adam.builder().learningRate(0.00002).beta1(0.5).build();//AdaGrad.builder().learningRate(0.01).build();//AdaGrad will be updating our network for us, using momentum and the like

    //Associated variables:
    private static JFrame frame;
    private static JPanel panel;
    private static int rows = 2; //How many rows of images/graphs/etc will be displayed?
    private static int columns = 2; //how many columns will be displayed?
    private static int scale = 5;


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




        auto.setListeners(new PerformanceListener(timeBetweenDisplays*128, true));
        System.out.println(" success");

        //Load the data
        System.out.print("Loading our data...");
        File dataLocation = new File(dataSetImageLocation);//where is the dataset of images
        float[][][] completeData =  getData(dataLocation);;
        float[][] dataSetSmalls = completeData[0];
        float[][] dataSetFulls = completeData[1];

        //the inputs are dataArray, and it is supposed to output dataArray
        System.out.println(" success");

        //set up some variables to record stuff
        int epoch = 0;

        System.out.println("Initiating training!");

        long lastTime = System.nanoTime();
        long timeSinceDisplay = 0;
        long timeSinceSave = 0;
        boolean displayThisEpoch;// shall we display the latest generation?
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
           //float[][] thisEpochsData = new float[batchSize][width*height*3]; @TODO COLOR OPTIONS HERE
            float[][] inputsThisEpoch = new float[batchSize][smallW*smallH];
            float[][] outputsThisEpoch = new float[batchSize][bigW*bigH];

            INDArray NDIn = null, NDOut = null;
            for (int i = 0; i < batchSize; i++)
            {
                int spot = (int)(Math.random()*dataSetFulls.length);
                inputsThisEpoch[i] = dataSetSmalls[spot];
                outputsThisEpoch[i] = dataSetFulls[spot];

                 NDIn = new NDArray(inputsThisEpoch);
                 NDOut = new NDArray(outputsThisEpoch);

                thisEpochsDataSet = new DataSet( NDIn, NDOut);
            }
            auto.fit(thisEpochsDataSet);//training here






            if (displayThisEpoch)
            {
                //Visualize all the stuff
                System.out.println("Visualizing:");
                INDArray[] imagesToDisplay = new INDArray[12];// 8 images

                System.out.print("    Getting Sample Images...");


                //4 enlargements
                for (int i = 0; i < 4; i++)
                {
                    imagesToDisplay[3*i] = NDIn.getRows(i);
                    imagesToDisplay[3*i+1] = auto.activateSelectedLayers(0, auto.getLayers().length - 1, NDIn.getRow(i));
                    imagesToDisplay[3*i+2] = NDOut.getRows(i);

                    }

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
                //.layer(0, new DenseLayer.Builder().nIn(width*height*3).nOut(middleLayerNeuronNumber).activation(Activation.LEAKYRELU).build())@TODO COLOR
                .layer(0, new DenseLayer.Builder().nIn(smallW*smallH).nOut(secondLayerNumbs).activation(Activation.LEAKYRELU).build())
                .layer(1, new DenseLayer.Builder().nIn(secondLayerNumbs).nOut(thirdLayerNumbs).activation(Activation.LEAKYRELU).build())
                .layer(2, new OutputLayer.Builder().nIn(thirdLayerNumbs).nOut(bigH*bigW).activation(Activation.LEAKYRELU)
                        .lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();

        /*
          .layer(0, new DenseLayer.Builder().nIn(width*height*3).nOut((width*height)/2).build())
                .layer(1, new DenseLayer.Builder().nIn((width*height)/2).nOut(dimensions).activation(Activation.TANH).build())
                .layer(2, new DenseLayer.Builder().nIn(dimensions).nOut((width*height)/2).build())//condensedLayerLocation
                .layer(3, new OutputLayer.Builder().nIn((width*height)/2).nOut(width*height*3)
                        .lossFunction(LossFunctions.LossFunction.MSE).build())
         */
    }



    ///////////////////////////////////////////////////////////////////////////////////////
    //Methods for Managing Data Here /////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////



    //Gets a float[][] array that will have all of your training data in it
    //
    //Takes in a DESIRED width and height (images will be scaled to this size)
    //And the path to where the training data is (where the images are)
    public static float[][][] getData(File whereTheImagesAre) throws IOException
    {


            //File folder = new File(System.getProperty("user.home") + "/Downloads/TrainingFiles/");

            //Gets all of the images as Files
            if (debug) System.out.println("Getting images from " + whereTheImagesAre.getAbsolutePath());
            List<File> listOfFiles =  new ArrayList<>();
            listOfFiles.addAll(FileUtils.listFiles(whereTheImagesAre,fileTypes,true));


            if (debug) System.out.println("Making the empty dataset:");

            System.out.println("Generating " + dataSetMultiplication + " data pieces per image");

            //this loop reads each file and adds it to the dataset.
            //spot keeps track of where in [][]dataSet to add the latest image


            //take all of the images and snip some parts of them out for use in the dataset
            ArrayList<BufferedImage> listOfSnippetImages = new ArrayList<>();
            ArrayList<BufferedImage> listOFSmallImages = new ArrayList<>();
            int spot = 0;

            for (File image : listOfFiles)
            {
                spot++;
                BufferedImage img = ImageIO.read(image);
                //Display progress
                    System.out.print(".");

                if (spot % 100 == 0)
                {
                    System.out.println();
                }



                for (int i = 0; i < dataSetMultiplication; i++)//get x snippets of the image from the image
                {
                    boolean gotAGoodImageYet = false;

                    for (int h = 0; h < triesPerImage && !gotAGoodImageYet; h++)
                    {
                        int randX = (int) (paddingAmount + Math.random() * (img.getWidth() - bigW-paddingAmount*2));
                        int randY = (int) (paddingAmount + Math.random() * (img.getHeight() - bigH-paddingAmount*2));

                        BufferedImage imagePart = img.getSubimage(randX, randY, bigW, bigH);

                        if (isABufferedImageGoodLooking(imagePart))
                        {
                            listOfSnippetImages.add(imagePart);
                      //      listOFSmallImages.add(scaleImage(imagePart,(double)(smallW)/bigH,(double)(smallW)/bigW));
                            listOFSmallImages.add(scaleImage(img.getSubimage(randX-paddingAmount,randY-paddingAmount,bigW+paddingAmount*2,bigH+paddingAmount*2),(double)(smallW)/(bigW+paddingAmount*2),(double)(smallH)/(bigH+paddingAmount*2)));
                            gotAGoodImageYet = true;
                        }

                    }

                }

            }

        ArrayList<float[]> outputArrayList = new ArrayList<>();
        ArrayList<float[]> inputArrayList = new ArrayList<>();

            //for each snippet image, turn it into a part of the dataset
            for (int v = 0; v < listOfSnippetImages.size(); v++)
            {
                BufferedImage img = listOfSnippetImages.get(v);
                BufferedImage downSized = listOFSmallImages.get(v);

                  //  float[] imageFloats = new float[width * height * 3];//creates an empty array for our latest image@TODO COLOR OPTIONS HERE
                float[] bigImageFloats = new float[bigH * bigW];//creates an empty array for our latest image
                float[] smallImageFloats = new float[smallH*smallW];

                    //get all the actual values from the image
                    for (int i = 0; i < img.getHeight() * img.getWidth(); i++)
                    {

                        //figure out where from the image to get the color of
                        int x = i % img.getWidth();
                        int y = i / img.getHeight();
                        Color atPixel = new Color(img.getRGB(x, y));


                        //get the red green and blue values, add to our array
                        bigImageFloats[i] = atPixel.getRed() / 255f;
                        //imageFloats[i + width * height] = atPixel.getGreen() / 255f;@TODO COLOR OPTIONS HERE
                        //imageFloats[i + width * height * 2] = atPixel.getBlue() / 255f;
                    }
                //get all the actual values from the image
                for (int i = 0; i < smallW*smallH; i++)
                {

                    //figure out where from the image to get the color of
                    int x = i % downSized.getWidth();
                    int y = i / downSized.getHeight();
                    Color atPixel = new Color(downSized.getRGB(x, y));


                    //get the red green and blue values, add to our array
                    smallImageFloats[i] = atPixel.getRed() / 255f;
                    //imageFloats[i + width * height] = atPixel.getGreen() / 255f;@TODO COLOR OPTIONS HERE
                    //imageFloats[i + width * height * 2] = atPixel.getBlue() / 255f;
                }
                outputArrayList.add(bigImageFloats);
                inputArrayList.add(smallImageFloats);
            }//end of adding all of the images

        float[][] outputFloatArray = new float[outputArrayList.size()][bigW*bigH];
        float[][] inputFloatArray = new float[inputArrayList.size()][smallH*smallW];
            for (int i = 0; i < outputArrayList.size(); i++)
            {
                outputFloatArray[i]= outputArrayList.get(i);
                inputFloatArray[i] = inputArrayList.get(i);
            }

            float[][][]dataSet = {inputFloatArray,outputFloatArray};
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
            panel.add(new JLabel(new ImageIcon(scaleImage(getImageRGB(images[i], (int) Math.sqrt(images[i].length()),(int) Math.sqrt(images[i].length())),scale,scale))));
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

        for (int i = 0; i < width*height;i++)//This turns the array into our image
        {
            //calculate where the x and y values should be on the actual image
            int x = i%width;
            int y = i/height;

            //find the reds, greens, and blues
            int red = (int) (255*inputArray.getFloat(i)); //get the red directly from the spot
            //int green = (int) (255*inputArray.getFloat(i+width*height)); //get the green from the spot 1/3 of the array forwards
            //int blue = (int) (255*inputArray.getFloat(i+width*height*2)); //get the blue from the spot 2/3 of the array forwards
            int green = red;
            int blue  = red;

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

    //tries to detect if a buffered image actually has writing on it (and isn't just blank)
    //tries to detect when there are two pixels of very different brightnesses near to eachother
    private static boolean isABufferedImageGoodLooking(BufferedImage input)
    {
        int iWidth = input.getWidth();
        int iHeight = input.getHeight();


        for (int y = 0; y < iHeight-5; y++)
        {
            for (int x = 0; x < iWidth-5; x++)
            {
                int atSpot = new Color(input.getRGB(x,y)).getRed();
                int atAFewAway = new Color(input.getRGB(x+4,y+4)).getRed();
                if (Math.abs(atSpot-atAFewAway)>53)//red = blue = green, as it is grayscale)//that's about the right ink color
                {
                    return true;
                }
            }
        }

        return false;
    }


    public static BufferedImage deepCopy(BufferedImage bi) {
        ColorModel cm = bi.getColorModel();
        boolean isAlphaPremultiplied = cm.isAlphaPremultiplied();
        WritableRaster raster = bi.copyData(bi.getRaster().createCompatibleWritableRaster());
        return new BufferedImage(cm, raster, isAlphaPremultiplied, null);
    }
}//The end!