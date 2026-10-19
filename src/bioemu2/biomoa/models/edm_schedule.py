"""EDM noise schedule for scaling model inputs and outputs (Karras et al.)."""

import torch


class EDMSchedule:
    def __init__(
        self,
        sigma_min: float = 0.0004,
        sigma_max: float = 160.0,
        sigma_data: float = 16.0,  # angstrom
    ):
        """
        EDM schedule for scaling model inputs and outputs according to Karras et al.

        Args:
            sigma_min: Minimum value for sigma. dimensionless.
            sigma_max: Maximal value for sigma. dimensionless.
            sigma_data: Standard deviation scale of data. unit: angstrom

        References:
            Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of
            diffusion-based generative models. Advances in neural information processing systems, 35,
            26565-26577.
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute `c_skip` for scaling the noisy coordinates when constructing the denoised ones
        (see Tab 1 in Karras et al):

            coords_denoised = c_skip * coords_noisy + c_out * model_output

        Args:
            sigma: Sigmas for which current scaling should be computed.

        Returns:
            Scaling value based on sigmas.
        """
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute `c_out` for scaling the model output when constructing the denoised coordinates
        (see Tab 1 in Karras et al):

            coords_denoised = c_skip * coords_noisy + c_out * model_output

        Args:
            sigma: Sigmas for which current scaling should be computed.

        Returns:
            Scaling value based on sigmas.
        """
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:  # 1/angstrom
        """
        Compute `c_in` for scaling the coordinates passed to the score model (see Tab 1 in Karras
        et al):

            coords_input = c_in * coords

        Args:
            sigma: Sigmas for which current scaling should be computed.

        Returns:
            Scaling value based on sigmas.
        """
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute noise conditioning. This is done similar to Karras et el., with the additional
        division by sigma_data (see AF3 SI).

        Args:
            sigma: Sigmas for which current scaling should be computed.

        Returns:
            Noise conditioning.
        """
        return torch.log(sigma / self.sigma_data) * 0.25
