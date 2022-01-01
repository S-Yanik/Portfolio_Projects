import com.google.common.util.concurrent.AtomicDouble;
import com.google.common.util.concurrent.AtomicDoubleArray;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class AutoSean
{

    //load the nets
    //display image
    //sliders update the image
    public static MultiLayerNetwork autoEncoder = null;
    public static int dimensions = 90;
    public static int bottleneckLocation = 2;

    public static AtomicDoubleArray values = new AtomicDoubleArray(dimensions);
    public static BufferedImage seanImage = null;
    public static int width = 68;
    public static int height = 68;
    public static int scale = 5;
    public static int sliderPrecisionLevel = 5; //how many tick marks per 1 unit? if it is 5, the range of possible
    // inputs for your values is 0, 0.2, 0.4, 0.6, 0.8, 1, etc
    public static SeanPanel seanPanel = null;

    public static JFrame frame;
    public static JSlider[] sliderArray = new JSlider[dimensions];
    public static JSlider insanitySlider = null;
    public static AtomicDouble insanityRating = new AtomicDouble(1);//regular insanity
    public static AtomicDouble mutationRate = new AtomicDouble(14 / 240d);
    public static AtomicBoolean monkeyMode = new AtomicBoolean(false);
    public static AtomicInteger mutationPower = new AtomicInteger(2);

    public static double[][] standardDeviationsAndMeans =
            {
                    {1.3184291,  -1.9635012},
                    {1.1143454,  1.0607489},
                    {1.2812743,  -2.122794},
                    {1.3293965,  1.6425794},
                    {1.1263304,  -3.2390466},
                    {1.6874282,  0.06793474},
                    {1.716432,  0.37785536},
                    {1.0864205,  1.6584553},
                    {1.4057993,  2.780536},
                    {1.2811594,  1.4840785},
                    {1.3185261,  1.6291887},
                    {1.73875,  -0.12967883},
                    {1.1510271,  -1.6093639},
                    {1.3526037,  -1.6212988},
                    {1.2437602,  2.3040082},
                    {1.2130065,  -0.90853804},
                    {1.2880337,  -0.033746764},
                    {1.1923515,  -2.2381063},
                    {1.1679782,  1.455376},
                    {1.6408468,  1.0245857},
                    {1.3102198,  -0.99530506},
                    {1.1510891,  -0.7344639},
                    {1.5110672,  2.505438},
                    {1.2065122,  -2.230175},
                    {1.4614555,  -2.9804316},
                    {1.5096477,  -4.342641},
                    {1.1449405,  -2.447658},
                    {1.0552062,  1.6544958},
                    {1.1828862,  -0.479899},
                    {1.1497827,  -1.0687492},
                    {1.0867883,  1.8851519},
                    {1.5342823,  0.29908672},
                    {1.171479,  1.4958943},
                    {1.6400094,  0.31288353},
                    {1.290808,  1.1482807},
                    {1.0930868,  -0.5246228},
                    {1.3630533,  1.7202567},
                    {1.1733103,  1.8254694},
                    {1.0152318,  -1.3248806},
                    {1.6251496,  1.1055925},
                    {1.3398885,  3.936369},
                    {1.1295586,  -0.6981749},
                    {1.3908918,  0.60767627},
                    {1.3754667,  3.127226},
                    {1.2303625,  1.8460565},
                    {1.2198272,  2.9385817},
                    {1.3069615,  -2.439613},
                    {1.2440125,  1.6893052},
                    {1.518797,  0.16196282},
                    {1.1206629,  -2.9032397},
                    {1.552014,  -0.6720307},
                    {1.1646677,  0.5692929},
                    {1.4481382,  -0.18407293},
                    {1.3421469,  1.482679},
                    {1.1233813,  1.0214801},
                    {1.1891111,  0.5162861},
                    {1.5095499,  -0.6994655},
                    {1.0447835,  0.86065024},
                    {1.0893226,  -1.9598515},
                    {1.4958693,  -0.46305338},
                    {1.0425583,  -1.6664437},
                    {1.5003065,  -1.3227804},
                    {1.4937384,  0.43166125},
                    {1.4257604,  -0.48177853},
                    {1.0815598,  -0.806296},
                    {1.2310427,  1.3988352},
                    {1.6545726,  0.9063628},
                    {1.2709321,  0.23611228},
                    {1.2667675,  2.6915317},
                    {1.3071702,  -0.27117652},
                    {1.3598092,  -2.8866823},
                    {1.2882001,  -0.69540447},
                    {1.3058296,  -0.431945},
                    {1.9985176,  -1.4783019},
                    {1.3051057,  -1.5100087},
                    {1.2864586,  -0.8126242},
                    {1.0502341,  -2.0869555},
                    {1.3129756,  -0.57569677},
                    {1.4080464,  -1.6524855},
                    {1.4091825,  0.9404328},
                    {1.1260767,  -2.8129675},
                    {1.3875562,  -2.695458},
                    {1.6671637,  2.1589787},
                    {1.0697349,  -1.0930796},
                    {0.98949546,  1.2596153},
                    {1.210892,  -2.6676776},
                    {0.87228537,  1.6205815},
                    {1.2850802,  -1.406912},
                    {1.7240516,  -0.25950402},
                    {1.145118,  -2.7420847},
            };
    //a real image
    public static double[] sean0 = new double[]
            {
                    -2.4484, 2.4448, -3.2949, 0.8277, -4.5863, 2.5832, 1.5892, 1.6520, 5.4357, 1.9111, 2.4030, 0.7830, -2.7270, -2.8080, 1.9690, -1.3543, -1.3727, -2.6132, 1.5322, -1.2006, -3.9693, -1.3294, 4.5302, -2.4329, -2.8673, -4.9071, -3.2969, 2.1959, 0.6770, -1.7281, 2.4123, 1.0444, 1.8989, 0.4017, 1.7796, -1.4142, 3.6477, 2.9698, -3.3193, 0.0221, 3.8160, -0.9987, 2.4363, 5.4309, 2.4856, 2.0191, -3.1121, 1.8429, 0.6088, -2.5978, -0.3863, 2.6330, -1.2107, 3.0740, 1.0169, 3.1234, -0.5487, -0.0952, -1.5298, -1.3369, -1.9850, -2.4304, -1.6878, -0.7322, -2.3171, 2.2275, 1.9714, 0.0994, 4.2339, -0.9369, -4.7087, -3.0567, -0.1104, -2.6522, -2.8349, -0.5296, -1.8829, -1.0663, -0.4850, 0.1464, -2.1852, -3.1295, 0.3726, -0.8927, 2.0474, -2.6316, 2.2884, -1.8677, -3.2238, -2.8776
            };


    public static void main(String args[]) throws IOException
    {
        //set up the slider values (min, max, starting)

        for (int i = 0; i < dimensions; i++)
        {
            sean0[i] = standardDeviationsAndMeans[i][1];
        }


        //load in the brain
        autoEncoder = ModelSerializer.restoreMultiLayerNetwork(System.getProperty("user.home") + "/Downloads/" + "autoSave" + ".zip");

        //set the values to the defaults
        for (int i = 0; i < values.length(); i++)
        {
            values.set(i, sliderPrecisionLevel * getDefaultValue(i));
        }


        seanPanel = new SeanPanel();

        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        int screenWide = screenSize.width * 3 / 4;
        int screenHigh = screenSize.height * 3 / 4;
        frame = new JFrame();
        frame.setPreferredSize(new Dimension(screenWide, screenHigh));
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setAlwaysOnTop(true);//move it on top (later in the code, this is disabled)
        frame.setResizable(true);//Sure, why not resize it
        frame.setVisible(true);
        frame.setLocation(20, 20);//move it over just a little bit, from the top left


        frame.getContentPane().add(seanPanel);

        addButtons();

        seanImage = new BufferedImage(width * scale, height * scale, BufferedImage.TYPE_INT_RGB);
        updateImage();


        frame.pack();//pack up the frame
        frame.setAlwaysOnTop(false);//make it so that the frame is not locked on top anymore


        while (true)
        {

            updateImage();

            seanPanel.repaint();

            if (monkeyMode.get())
            {
                for (int i = 0; i < dimensions; i++)
                {
                    int value = sliderArray[i].getValue();
                    if (Math.random() < mutationRate.doubleValue())
                    {

                        if (Math.random() > 0.5)
                        {
                            sliderArray[i].setValue(value + mutationPower.get());
                        } else
                        {
                            sliderArray[i].setValue(value - mutationPower.get());
                        }


                    }

                }
            }

        }


    }

    public static void addButtons()
    {
        JPanel p = new JPanel();
        JPanel southPanel = new JPanel();

        JButton saveButton = new JButton("Save Image");
        saveButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                synchronized (seanImage)
                {
                    BufferedImage toSave = seanImage;
                    String name = Math.random() + ".PNG";
                    File location = new File(System.getProperty("user.home") + "/Desktop/" + name);
                    try
                    {
                        System.out.println("saving");
                        ImageIO.write(toSave, "PNG", location);
                        System.out.print("Saved");
                    } catch (IOException ex)
                    {
                        ex.printStackTrace();
                    }
                }
            }
        });
        p.add(saveButton);


        //slider buttons
        southPanel.add(new JLabel("Aspects:"));

        //make the twelve or 18 or whatever the number of dimensions is sliders
        for (int i = 0; i < dimensions; i++)
        {
            final int finalI = i;
            final JSlider dimensionSlider = new JSlider(JSlider.VERTICAL, (int) (sliderPrecisionLevel * getMinimumValue(finalI)),
                    (int) (sliderPrecisionLevel * getMaximumValue(finalI)),
                    (int) (sliderPrecisionLevel * getDefaultValue(finalI)));

            dimensionSlider.addChangeListener(new ChangeListener()
            {
                public void stateChanged(ChangeEvent e)//when clicked...
                {
                    double calculatedValue = insanityRating.doubleValue() * dimensionSlider.getValue();
                    values.set(finalI, calculatedValue);

                    //( getMaximumValue(finalI) + Math.abs(getMinimumValue(finalI)))*(dimensionSlider.getValue()/100d) - Math.abs(getMinimumValue(finalI));
                }
            });
            dimensionSlider.setMajorTickSpacing(10);//visual look of the slider
            dimensionSlider.setMinorTickSpacing(5);
            dimensionSlider.setPaintTicks(true);


            sliderArray[i] = dimensionSlider;
            southPanel.add(dimensionSlider);//add the slider
        }

        //reset
        JButton b1 = new JButton("Reset");
        b1.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                for (int i = 0; i < dimensions; i++)
                {
                    // int percentValue = (int) (100*calcPercentBetween(getMinimumValue(i),getMaximumValue(i),getDefaultValue(i)));

                    sliderArray[i].setValue((int) (sliderPrecisionLevel * getDefaultValue(i)));
                    insanitySlider.setValue(20);
                    updateImage();
                }
            }
        });
        p.add(b1);


        //extreme scalar
        p.add(new JLabel("Over Scaler"));

        final JSlider stenchSlider = new JSlider(JSlider.VERTICAL, 0, 30, 20);
        stenchSlider.addChangeListener(new ChangeListener()
        {
            public void stateChanged(ChangeEvent e)//when clicked
            {
                insanityRating.set(stenchSlider.getValue() / 20d);
                for (int i = 0; i < sliderArray.length; i++)
                {
                    sliderArray[i].setValue(sliderArray[i].getValue() + 1);
                    sliderArray[i].setValue(sliderArray[i].getValue() - 1);
                }
                updateImage();
            }
        });
        stenchSlider.setMajorTickSpacing(5);
        stenchSlider.setPaintTicks(true);
        insanitySlider = stenchSlider;
        p.add(stenchSlider);
        //end of extreme scalar

        //Mutation button
        JButton b2 = new JButton("Toggle Mutations");
        b2.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)//when clicked
            {
                monkeyMode.set(!monkeyMode.get());
                System.out.println(monkeyMode.get());

            }
        });
        p.add(b2);
        //end of mutation button

        p.add(new JLabel("Mutation Speed"));

        final JSlider mutationSlider = new JSlider(JSlider.VERTICAL, 1, 60, 14);
        mutationSlider.addChangeListener(new ChangeListener()
        {
            public void stateChanged(ChangeEvent e)//when clicked
            {
                mutationRate.set(mutationSlider.getValue() / 240d);

            }
        });
        mutationSlider.setMajorTickSpacing(5);
        mutationSlider.setPaintTicks(true);
        p.add(mutationSlider);

        p.add(new JLabel("Mutation Strength"));

        final JSlider mutationPowerSlider = new JSlider(JSlider.VERTICAL, 1, 10, 2);
        mutationPowerSlider.addChangeListener(new ChangeListener()
        {
            public void stateChanged(ChangeEvent e)//when clicked
            {
                mutationPower.set(mutationPowerSlider.getValue());

            }
        });
        mutationPowerSlider.setMajorTickSpacing(2);
        mutationPowerSlider.setPaintTicks(true);
        p.add(mutationPowerSlider);

        southPanel.setLayout(new GridLayout(2, 40, 0, 0));

        frame.getContentPane().add(p, BorderLayout.EAST);
        frame.getContentPane().add(southPanel, BorderLayout.SOUTH);
    }


    public static void updateImage()
    {
        synchronized (seanImage)
        {
            float[] doubleToFloats = new float[dimensions];
            for (int i = 0; i < dimensions; i++)
            {
                doubleToFloats[i] = (float) (values.get(i) / sliderPrecisionLevel);
            }
            INDArray input = new NDArray(doubleToFloats);

            INDArray generatedSean = autoEncoder.activateSelectedLayers(bottleneckLocation, autoEncoder.getLayers().length - 1, input);

            seanImage = scaleImage(getImageRGB(generatedSean, width, height), scale, scale);
        }
    }

    private static double getDefaultValue(int i)
    {
        return sean0[i];
        // return standardDeviationsAndMeans[i][1];
    }

    private static double getMinimumValue(int i)
    {
        // return sean0[i];
        return Math.min(standardDeviationsAndMeans[i][1] - standardDeviationsAndMeans[i][0] * 1.5, getDefaultValue(i));
    }

    private static double getMaximumValue(int i)
    {
        // return sean0[i];
        return Math.max(standardDeviationsAndMeans[i][1] + standardDeviationsAndMeans[i][0] * 1.5, getDefaultValue(i));
    }

    private static BufferedImage getImageRGB(INDArray inputArray, int width, int height)
    {
        //the image to return (B uffered I mage --> bi). Sorry, I guess that's pretty obvious.
        BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        //go through the 1/3 of the INDArray (the array is width * height for red, green and blue, so just width
        // times height is 1/3 of it)
        for (int i = 0; i < width * height; i++)//This turns the array into our image
        {
            //calculate where the x and y values should be on the actual image
            int x = i % width;
            int y = i / width;

            //find the reds, greens, and blues
            int red = (int) (255 * inputArray.getFloat(i)); //get the red directly from the spot
            int green = (int) (255 * inputArray.getFloat(i + width * height)); //get the green from the spot 1/3 of the array forwards
            int blue = (int) (255 * inputArray.getFloat(i + width * height * 2)); //get the blue from the spot 2/3 of the array forwards

            //There is a possibility that the AI has generated impossible colors
            //(IE darker than black (RGB = -20, -5, 0, for example) or brighter than white
            //clamp these values to get something decent (clamping forces the values to be between the min and max)
            red = clamp(red, 0, 255);
            green = clamp(green, 0, 255);
            blue = clamp(blue, 0, 255);

            //get the color for the pixel (stored as an int)
            int colorOfPixel = new Color(red, green, blue).getRGB();

            //set the color at the spot
            bi.setRGB(x, y, colorOfPixel);

        }//end of coloring in our buffered image. It is now correctly made

        return bi;
    }//end of getImageRGB

    //Scales a buffered image
    private static BufferedImage scaleImage(BufferedImage image, double xScale, double yScale)
    {
        int width = image.getWidth();
        int height = image.getHeight();

        Image imageScaled = image.getScaledInstance((int) (xScale * width), (int) (yScale * height), Image.SCALE_REPLICATE);

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
        return Math.max(min, Math.min(max, value));
    }

    //returns a percentage between 0.0 and 1.0 (0.55 --> 55%)
    public static double calcPercentBetween(double minimum, double maximum, double value)
    {
        //gives you the percent value is between the minimum and maximum (as decimal 0.0 --> 1.0)

        return (value - minimum) / (maximum - minimum);

    }

    public static class SeanPanel extends JPanel
    {
        public void paint(Graphics g)
        {
            g.setColor(Color.DARK_GRAY);
            g.fillRect(0, 0, width * scale, height * scale);
            g.drawImage(seanImage, 0, 0, null);
        }
    }

}
