<?php

namespace PdfOxide\Managers;

class EncryptionSettings {
    public $algorithm;
    public $userPassword;
    public $ownerPassword;
    public $allowPrinting;
    public $allowCopying;
    public $allowModification;

    public function __construct($algorithm, $userPassword, $ownerPassword, $allowPrinting, $allowCopying, $allowModification) {
        $this->algorithm = $algorithm;
        $this->userPassword = $userPassword;
        $this->ownerPassword = $ownerPassword;
        $this->allowPrinting = $allowPrinting;
        $this->allowCopying = $allowCopying;
        $this->allowModification = $allowModification;
    }
}
