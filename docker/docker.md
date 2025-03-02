# Setting before build the image

## NVIDIA Container Runtime

You need to use **nvidia-container-runtime** as explained in the documentation:

1. **Install nvidia-container-runtime:**

    ```bash
    sudo apt-get install nvidia-container-runtime
    ```

2. **Edit or create the file `/etc/docker/daemon.json` with the following content:**

    ```json
    {
        "runtimes": {
            "nvidia": {
                "path": "/usr/bin/nvidia-container-runtime",
                "runtimeArgs": []
            }
        },
        "default-runtime": "nvidia"
    }
    ```

3. **Restart the Docker daemon:**

    ```bash
    sudo systemctl restart docker
    ```

## X11 Configuration for GUI Applications

If you plan to run graphical applications inside the container, you must allow local connections to your host's X server. To do this, execute the following command before running your container:

```bash
xhost +local:
```