import javax.swing.event.ChangeListener;

public abstract class StoppableListener implements ChangeListener
{
        public boolean active = true;

        public void setActive(boolean active)
        {
            this.active = active;
        }


}