import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;


public class ArtGan
{



    private static int timeBetweenDisplays = 4; //in seconds
    private static int batchSize = 5;

    private static int timeBetweenSaves = 450;//in seconds
    private static boolean loading = false;
    private static boolean saving = true;
    private static String ganSaveName = "ganSaveArt";
    private static String disSaveName = "disSaveArt";

    //Associated Variables for dataset
    private static int artWidthFromData = 100;
    private static int artHeightFromData = 100;
    private static int artWidth = 40;
    private static int artHeight = 40;
    private static int dataSetMultiplication = 1;
    private static int triesPerImage = 5;


    private static String dataSetImageLocation = "C:\\Users\\Sean\\Downloads\\ExportImages";
    private static String[] fileTypes = new String[]{ "png"}; //what extensions can your files have?



    private static int ganRandInputs = 64;
    private static int seed = 735;
    private static final IUpdater UPDATER = Adam.builder().learningRate(0.0002).beta1(0.5).build();//AdaGrad.builder().learningRate(0.01).build();//AdaGrad will be updating our network for us, using momentum and the like
    private static final IUpdater UPDATER_ZERO = Sgd.builder().learningRate(0.0).build();


    private static JFrame frame;
    private static JPanel panel;
    private static int rows = 2; //How many rows of images/graphs/etc will be displayed?
    private static int columns = 2; //how many columns will be displayed?
    private static int scale = 2;


    public static void main(String args[]) throws IOException
    {
        MultiLayerNetwork gan = null;
        MultiLayerNetwork dis = null;
        MultiLayerNetwork gen = null;
        if (!loading)
        {
            System.out.print("Creating the nets...");
             gen = new MultiLayerNetwork(generator());
             dis = new MultiLayerNetwork(discriminator());
             gan = new MultiLayerNetwork(gan());
            gan.init();
            dis.init();
            gen.init();

              copyParams(gen,dis,gan);
        }else{
            System.out.println("Restoring the nets...");
            gan = ModelSerializer.restoreMultiLayerNetwork(System.getProperty("user.home") + "/Downloads/networkSaves/"+ ganSaveName +".zip");
            dis = ModelSerializer.restoreMultiLayerNetwork(System.getProperty("user.home") + "/Downloads/networkSaves/"+ disSaveName +".zip");
        }
        gan.setListeners(new PerformanceListener(timeBetweenDisplays, true));
        dis.setListeners(new PerformanceListener(timeBetweenDisplays, true));
        System.out.println(" success");






        //Load the data
        System.out.print("Loading our data...");
        File dataLocation = new File(dataSetImageLocation);//where is the dataset of images
        float[][] data = getData(dataLocation);
        System.out.println(" success");






        System.out.println("Initiating training!");
        int epoch = 0;
        long lastTime = System.nanoTime();
        long timeSinceDisplay = 0;
        long timeSinceSave = 0;
        boolean displayThisEpoch;// shall we display the latest generation?




        while (true)
        {
            {
            epoch++;

            displayThisEpoch = false;

                //calculate time passed
                long thisTime = System.nanoTime();
                long deltaTime = thisTime - lastTime;
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
                        ModelSerializer.writeModel(gan, new File(downloads + "/networkSaves/" + ganSaveName + ".zip"), true);
                        ModelSerializer.writeModel(dis, new File(downloads + "/networkSaves/" + disSaveName + ".zip"),true);
                        System.out.println("\n\n\n\n\n\n\n\n\nSAVED!");
                    }
                }

            if (displayThisEpoch) System.out.print("Epoch: " + epoch + ". Training!");
            //get data
            }//display and save


            float[][] realImages = new float[batchSize][artWidth*artHeight];
            INDArray fakeNDArry = gan.activateSelectedLayers(0,gen.getLayers().length - 1, Nd4j.rand(new int[]{batchSize, ganRandInputs}));
            for (int i = 0; i < batchSize; i++)
            {
                int spot = (int)(Math.random()*data.length);
                realImages[i] = data[spot];
            }

            INDArray realNDArray = new NDArray(realImages);

            DataSet realSet = new DataSet(realNDArray, Nd4j.zeros(batchSize, 1));
            DataSet fakeSet = new DataSet(fakeNDArry, Nd4j.ones(batchSize, 1));

            DataSet thisEpochsData = DataSet.merge(Arrays.asList(realSet, fakeSet));


            dis.fit(thisEpochsData);

            updateGan(gen,dis,gan);

            gan.fit(new DataSet(Nd4j.rand(new int[]{batchSize, ganRandInputs}), Nd4j.zeros(batchSize, 1)));
           // if (gan.score() > 4 )
           // {
          //      System.out.println("boost");
         //       gan.fit(new DataSet(Nd4j.rand(new int[]{batchSize, ganRandInputs}), Nd4j.zeros(batchSize, 1)));
          //      gan.fit(new DataSet(Nd4j.rand(new int[]{batchSize, ganRandInputs}), Nd4j.zeros(batchSize, 1)));
         //   }


            if (displayThisEpoch)
            {
                //Visualize all the stuff
                System.out.println("Visualizing:");
                INDArray[] imagesToDisplay = new INDArray[4];// 8 images
                System.out.print("    Getting Sample Images...");
                for (int i = 0; i < 4; i++)
                {
                    if(i%2 == 0)
                    {
                        imagesToDisplay[i] = new NDArray(realImages[i]);
                    }
                    else
                    {
                        imagesToDisplay[i] = fakeNDArry.getRows(i);//@Todo maybe this should be getColumns?
                    }

                }
                display(imagesToDisplay);
            }


        }//infinite learning loop
    }//end of main

    ///////////////////////////////////////////////////////////////////////////////////////
    //The Setup for our network goes here /////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    private static Layer[] genLayers()
    {
        return new Layer[]{
                new DenseLayer.Builder().nIn(ganRandInputs).nOut(128).weightInit(WeightInit.RELU).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DenseLayer.Builder().nIn(128).nOut(256).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DenseLayer.Builder().nIn(256).nOut(512).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DenseLayer.Builder().nIn(512).nOut(artWidth*artHeight).activation(Activation.TANH).build()
        };
    }


    private static MultiLayerConfiguration generator()
    {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(UPDATER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                //.l2(L2)
                .weightInit(WeightInit.RELU)
                .activation(Activation.IDENTITY)
                .list(genLayers())
                .build();

        return conf;
    }

    private static Layer[] disLayers(IUpdater updater)
    {
        return new Layer[]{
                new DenseLayer.Builder().nIn(artWidth*artHeight).nOut(1000).updater(updater).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DropoutLayer.Builder(1 - 0.5).build(),
                new DenseLayer.Builder().nIn(1000).nOut(512).updater(updater).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
             //   new DropoutLayer.Builder(1 - 0.5).build(),
                new DenseLayer.Builder().nIn(512).nOut(256).updater(updater).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
               // new DropoutLayer.Builder(1 - 0.5).build(),
                new OutputLayer.Builder(LossFunctions.LossFunction.XENT).nIn(256).nOut(1).activation(Activation.SIGMOID).updater(updater).build()
        };
    }

    private static MultiLayerConfiguration discriminator()
    {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(UPDATER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                //.l2(L2)
                .weightInit(WeightInit.RELU)
                .activation(Activation.IDENTITY)
                .list(disLayers(UPDATER))
                .build();

        return conf;
    }

    private static MultiLayerConfiguration gan()
    {
        Layer[] genLayers = genLayers();
        Layer[] disLayers = disLayers(UPDATER_ZERO); // Freeze discriminator layers in combined network.
        Layer[] layers = ArrayUtils.addAll(genLayers, disLayers);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(UPDATER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                //.l2(L2)
                .weightInit(WeightInit.RELU)
                .activation(Activation.IDENTITY)
                .list(layers)
                .build();

        return conf;
    }









    public static float[][] getData(File whereTheImagesAre) throws IOException
    {


           System.out.println("Getting images from " + whereTheImagesAre.getAbsolutePath());
            List<File> listOfFiles =  new ArrayList<>();
            listOfFiles.addAll(FileUtils.listFiles(whereTheImagesAre,fileTypes,true));
            System.out.println("Making the empty dataset:");
            System.out.println("Generating " + dataSetMultiplication + " data pieces per image");






            ArrayList<BufferedImage> listOfImages = new ArrayList<>();
            int spot = 0;
            for (File image : listOfFiles)
            {
                spot++;
                System.out.print(".");
                if (spot % 100 == 0) System.out.println();




                BufferedImage img = ImageIO.read(image);
                if (img.getHeight() < artHeightFromData || img.getWidth() < artWidthFromData)
                {
                    System.out.println("Image " +  spot + " was too small.");
                }
                else
                {
                    for (int i = 0; i < dataSetMultiplication; i++)//get x snippets of the image from the image
                    {
                        boolean gotAGoodImageYet = false;

                        for (int h = 0; h < triesPerImage && !gotAGoodImageYet; h++)
                        {
                            int randX = (int) (Math.random() * (img.getWidth() - artWidthFromData));
                            int randY = (int) (Math.random() * (img.getHeight() - artHeightFromData));

                            BufferedImage imagePart = img.getSubimage(randX, randY, artWidthFromData, artHeightFromData);
                            if (!isABufferedImageGoodLooking(imagePart))
                            {
                                imagePart = scaleImage(imagePart,(double)(artWidth)/artWidthFromData,
                                        (double)(artHeight)/artHeightFromData);
                                //@TODO add the mirrors, and rotated mirrors, to multiply the dataset size by 8
                                listOfImages.add(imagePart);
                                gotAGoodImageYet = true;
                            }
                        }
                    }
                }
            }

        ArrayList<float[]> floatArrayList = new ArrayList<>();


            for (int v = 0; v < listOfImages.size(); v++)
            {
                BufferedImage img = listOfImages.get(v);


                float[] imageFloats = new float[artWidth * artHeight];//creates an empty array for our latest image

                    for (int i = 0; i < img.getHeight() * img.getWidth(); i++)
                    {
                        int x = i % img.getWidth();
                        int y = i / img.getHeight();
                        Color atPixel = new Color(img.getRGB(x, y));
                        imageFloats[i] = atPixel.getRed() / 255f;
                    }


                floatArrayList.add(imageFloats);

            }//end of adding all of the images

        float[][] floatArrayArray = new float[floatArrayList.size()][artHeight*artWidth];

            for (int i = 0; i < floatArrayList.size(); i++)
            {
                floatArrayArray[i]= floatArrayList.get(i);
            }

            return floatArrayArray;
    }












    //Displays what's new with our gan. Takes in a collection of INDArrays to turn into images, which it displays
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


        for (int y = 0; y < iHeight-4; y+=3)
        {
            for (int x = 0; x < iWidth-4; x+=3)
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
        boolean isAlphaPreMultiplied = cm.isAlphaPremultiplied();
        WritableRaster raster = bi.copyData(bi.getRaster().createCompatibleWritableRaster());
        return new BufferedImage(cm, raster, isAlphaPreMultiplied, null);
    }


    //copies the random setups of the gen and dis to the gan network
    private static void copyParams(MultiLayerNetwork gen, MultiLayerNetwork dis, MultiLayerNetwork gan)
    {
        int genLayerCount = gen.getLayers().length;
        for (int i = 0; i < gan.getLayers().length; i++)
        {
            if (i < genLayerCount)
            {
                gen.getLayer(i).setParams(gan.getLayer(i).params());
            } else
            {
                dis.getLayer(i - genLayerCount).setParams(gan.getLayer(i).params());
            }
        }
    }


    private static void updateGan(MultiLayerNetwork gen, MultiLayerNetwork dis, MultiLayerNetwork gan)
    {
        int genLayerCount = gen.getLayers().length;
        for (int i = genLayerCount; i < gan.getLayers().length; i++)
        {
            gan.getLayer(i).setParams(dis.getLayer(i - genLayerCount).params());
        }


    }



}//The end!