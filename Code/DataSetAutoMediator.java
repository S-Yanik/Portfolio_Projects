import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class DataSetAutoMediator
{

    //Associated Variables:
    private static int width = 68;//input data will be scaled down to this width
    private static int height = 68;//by this height
    private static int dimensions = 90;
    private static int bottleNeckLayer = 2;
    private static String[] fileTypes = new String[]{ "jpg"}; //what extensions can your files have?
    private static boolean debug = true;



    public static void main(String args[]) throws IOException
    {
        MultiLayerNetwork autoEncoder = ModelSerializer.restoreMultiLayerNetwork(System.getProperty("user.home") + "/Downloads/" + "autoSave" + ".zip");

        File dataLocation = new File("C:\\Users\\Sean\\Downloads\\face dataset");//where is the dataset of images
        float[][]dataset = getData(width,height,dataLocation);

        float[][] resultsOf12 = new float[dimensions][dataset.length];

        do
        {
            System.out.println("Getting the condensed images");
            for (int i = 0; i < dataset.length; i++)
            {
                INDArray inputs = new NDArray(dataset[i]);
                INDArray outputsOf12 = autoEncoder.activateSelectedLayers(0, bottleNeckLayer - 1, inputs);

                System.out.println(outputsOf12);
                for (int j = 0; j < dimensions; j++)
                {
                    resultsOf12[j][i] = outputsOf12.getFloat(j);
                }
            }

            System.out.println("Analyzing data: (standard deviation, mean)");


            System.out.println("{");
            for (int i = 0; i < dimensions; i++)
            {
                float total = 0;
                int numberOfNumbers = dataset.length;

                for (int j = 0; j < resultsOf12[i].length; j++)
                {
                    total += resultsOf12[i][j];
                }
                float mean = total / numberOfNumbers;

                //calculate the variance
                total = 0;
                for (int j = 0; j < resultsOf12[i].length; j++)
                {
                    total += (float) Math.pow(mean - resultsOf12[i][j], 2);
                }

                float variance = total / numberOfNumbers;

                float standardDeviation = (float) Math.sqrt(variance);

                System.out.println("{" + standardDeviation + ",  " + mean + "},");
            }

            System.out.println("};");

        } while (false);

        INDArray inputs = new NDArray(dataset[0]);
        INDArray sean0 = autoEncoder.activateSelectedLayers(0,bottleNeckLayer-1,inputs);
                System.out.println("Sean 0:");
        System.out.println(sean0);



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
        float[][]dataSet = new float[(Math.min(28000,listOfFiles.size()))][width*height*3];
        //float[][]dataSet = new float[listOfFiles.size()][width*height];


        //this loop reads each file and adds it to the dataset.
        //spot keeps track of where in [][]dataSet to add the latest image
        int spot = 0;
        for (int f = 0; f < (Math.min(28000,listOfFiles.size())); f++)
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


}
