import org.apache.commons.io.FileUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dimensionalityreduction.PCA;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;

public class PCATest
{

    private static int width = 20;//input data will be scaled down to this width
    private static int height = 20;//by this height
    private static String[] fileTypes = new String[]{ "jpg"}; //what extensions can your files have?
    private static String dataSetLocation = "C:\\Users\\Sean\\Downloads\\face dataset";
    private static int dataSetSize = 800;
    //Associated variables for visualisation:
    private static JFrame frame;
    private static JPanel panel;
    private static int rows = 2; //How many rows of images/graphs/etc will be displayed?
    private static int columns = 2; //how many columns will be displayed?
    private static int scale = 3;



    public static void main(String args[]) throws IOException
    {
        System.out.print("Loading our data...");
        File dataLocation = new File(dataSetLocation);//where is the dataset of images
        float[][] dataSetFloats = getData(width, height, dataLocation);

        INDArray fullDataset = new NDArray(dataSetFloats);

        PCA pca = null;
        System.out.println("doing the pca thing");
        pca = new PCA(fullDataset);

        System.out.println("Alright, thank goodness, that's done.");


        INDArray eigenvalues = pca.getEigenvalues();
        float[] values = new float[1200];

        for (int i = 0; i < 1200; i++)
        {
        values[i] = eigenvalues.getFloat(i);
        }

        float[] clone = values.clone();

        Arrays.sort(values);

        float[] highest80 = new float[80];
        for (int i = 0; i < 80; i++)
        {
            highest80[i] = values[1199-i];
        }

        int[] locations = new int[80];
        for (int i = 0; i < 80; i++)
        {
            for (int j = 0; j < 1200; j++)
            {
                if (clone[j] == highest80[i])
                    locations[i] = j;
            }
        }






            Scanner kkb = new Scanner(System.in);
            while(kkb.nextLine() != "v")
            {
                INDArray[] imagesToDisplay = new INDArray[15];


                int locationInDataSet = 0;

                for (int i = 0; i < 12; i++)
                {
                    if (i%2 == 0)
                    {
                        locationInDataSet =  (int)(Math.random()*dataSetFloats.length);
                        imagesToDisplay[i] = new NDArray(dataSetFloats[locationInDataSet]);
                    }else
                    {
                        INDArray arr = pca.convertToComponents(imagesToDisplay[i-1]);
                        float[] flo = arr.toFloatVector();

                        for (int o = 0; o < 1200; o++)
                        {
                            boolean onList = false;
                            for (int g = 0; g < 80; g++)
                            {
                                if (locations[g] == o)
                                    onList = true;
                            }

                            if (!onList)
                                flo[o] = pca.getCovarianceMatrix().getFloat(o);
                        }

                        imagesToDisplay[i] = new NDArray(flo);
                    }
                }

                float[] finalOne = new float[1200];

                System.out.println(pca.getMean());
                for (int g = 0; g < 1200; g++)
                {
                    finalOne[g] = pca.getCovarianceMatrix().getFloat(g) + pca.getMean().getFloat(g);
                }

                imagesToDisplay[12] = new NDArray(finalOne);
                imagesToDisplay[13] = imagesToDisplay[12];
                imagesToDisplay[14] = imagesToDisplay[13];

                display(imagesToDisplay);
            }

    }


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


    public static float[][] getData(int width, int height, File whereTheImagesAre) throws IOException
    {


        //File folder = new File(System.getProperty("user.home") + "/Downloads/TrainingFiles/");

        //Gets all of the images as Files
        System.out.println("Getting images from " + whereTheImagesAre.getAbsolutePath());
        List<File> listOfFiles =  new ArrayList<>();
        listOfFiles.addAll(FileUtils.listFiles(whereTheImagesAre,fileTypes,true));



        //creates a nice, empty dataset with as many slots as there are files in our folder
        System.out.println("Making the empty dataset:");
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


    //clamps value between min and max
    public static int clamp(int value, int min, int max)
    {
        return Math.max(min,Math.min(max,value));
    }

}
