"""
Support for AzureSpeech STT.
"""
import logging
from typing import AsyncIterable
import async_timeout
import voluptuous as vol
from homeassistant.components.tts import CONF_LANG
from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    Provider,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
)
import homeassistant.helpers.config_validation as cv
import wave
import io
import openai
import azure.cognitiveservices.speech as speechsdk

_LOGGER = logging.getLogger(__name__)

CONF_API_KEY = 'api_key'
DEFAULT_LANG = 'en-US'

PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend({
    vol.Required(CONF_API_KEY): cv.string,
    vol.Required("region"): cv.string,
    vol.Optional(CONF_LANG, default=DEFAULT_LANG): cv.string,
})

async def async_get_engine(hass, config, discovery_info=None):
    """Set up Azure Speech STT speech component."""
    api_key = config[CONF_API_KEY]
    languages = config.get(CONF_LANG, DEFAULT_LANG)
    region = config.get('region')
    return AzureSpeechSTTProvider(hass, api_key, languages, region)

class AzureSpeechSTTProvider(Provider):
    """The Azure Speech STT API provider."""

    def __init__(self, hass, api_key, lang, region):
        """Initialize OpenAI STT provider."""
        self.hass = hass
        self._api_key = api_key
        self._language = lang

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return self._language.split(',')[0]

    @property
    def supported_languages(self) -> list[str]:
        """Return the list of supported languages."""
        # Ideally, this list should be dynamically fetched from OpenAI, if supported.
        return self._language.split(',')

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bitrates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported samplerates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        if metadata.format != AudioFormats.WAV:
            _LOGGER.error("Unsupported audio format: %s", metadata.format)
            return SpeechResult("", SpeechResultState.ERROR)
        
        data = b''

        async for chunk in stream:
            data += chunk

        wav_stream = io.BytesIO()

        with wave.open(wav_stream, 'w') as wav_file:
            wav_file.setnchannels(metadata.channel)
            wav_file.setsampwidth(metadata.bit_rate // 8)
            wav_file.setframerate(metadata.sample_rate)

            wav_file.writeframes(data)
            
        wav_stream.seek(0)

        config = speechsdk.SpeechConfig(subscription=self._api_key, region=self._region)

        reg = speechsdk.SpeechRecognizer(speech_config=config, audio_config=speechsdk.AudioConfig(stream=wav_stream), auto_detect_source_language_config=speechsdk.AutoDetectSourceLanguageConfig(languages=self.supported_languages))

        async with async_timeout.timeout(20):
            result = await reg.recognize_once_async()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return SpeechResult(result.text, SpeechResultState.SUCCESS)
            else:
                return SpeechResult(str(result.reason), SpeechResultState.ERROR)