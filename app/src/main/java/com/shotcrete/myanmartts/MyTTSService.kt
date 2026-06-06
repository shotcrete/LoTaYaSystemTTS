package com.shotcrete.lotayatts

import android.content.Context
import android.media.AudioFormat
import android.os.Bundle
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.UtteranceProgressListener
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.HashMap
import java.util.Locale
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class MyTTSService : TextToSpeechService(), TextToSpeech.OnInitListener {

    private var ortEnv: OrtEnvironment? = null
    private var ortSessionMm: OrtSession? = null
    private var googleTts: TextToSpeech? = null
    private var isGoogleTtsReady = false

    // မြန်မာ Vocab Map (ခင်ဗျားပေးထားသည့်အတိုင်း)
    private val vocabMapMm = mapOf(
        '်' to 0L, 'ာ' to 1L, 'ု' to 2L, 'ိ' to 3L, 'း' to 4L, 'ေ' to 5L, 'သ' to 6L, 'က' to 7L,
        'င' to 8L, 'တ' to 9L, '့' to 10L, 'မ' to 11L, 'ြ' to 12L, 'ည' to 13L, 'ရ' to 14L, 'အ' to 15L,
        'န' to 16L, 'လ' to 17L, 'ှ' to 18L, 'ပ' to 19L, 'စ' to 20L, 'ခ' to 21L, 'ျ' to 22L, 'ူ' to 23L,
        'ွ' to 24L, 'ါ' to 25L, 'ထ' to 26L, 'ဖ' to 27L, 'ံ' to 28L, 'ယ' to 29L, 'ဆ' to 30L, 'ီ' to 31L,
        'ဲ' to 32L, 'ဟ' to 33L, 'ဘ' to 34L, 'ဝ' to 35L, '္' to 36L, 'ဉ' to 37L, 'ဤ' to 38L, 'ဇ' to 39L,
        'ဒ' to 40L, 'ဂ' to 41L, 'ဦ' to 42L, 'ဏ' to 43L, 'ဗ' to 44L, 'ဓ' to 45L, 'ဧ' to 46L, 'ဥ' to 47L,
        'ဩ' to 48L, 'ဌ' to 49L, 'ဋ' to 50L, '\'' to 51L, 'ဣ' to 52L, 'ဍ' to 53L, 'ဿ' to 54L, 'ဈ' to 55L,
        ' ' to 56L
    )

    override fun onCreate() {
        super.onCreate()
        // Fallback အတွက် Google TTS ကို နှိုးခြင်း
        googleTts = TextToSpeech(this, this)

        // ONNX Engine အား Background Thread တွင် ဆွဲဖွင့်ခြင်း
        Thread {
            try {
                ortEnv = OrtEnvironment.getEnvironment()
                val modelFileMm = File(cacheDir, "model.onnx")
                if (!modelFileMm.exists() || modelFileMm.length() < 10_000_000) {
                    if (modelFileMm.exists()) modelFileMm.delete()
                    copyAssetToFile("model.onnx", modelFileMm)
                }
                ortSessionMm = ortEnv?.createSession(modelFileMm.absolutePath)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }.start()
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = googleTts?.setLanguage(Locale.US)
            isGoogleTtsReady = !(result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED)
        } else {
            isGoogleTtsReady = false
        }
    }

    // --- Android System TTS အသိအမှတ်ပြုစေမည့် Engine ချိန်ညှိချက်များ ---
    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int {
        if (lang == "mya" || lang == "eng" || lang == "bur") {
            return TextToSpeech.LANG_COUNTRY_AVAILABLE
        }
        return TextToSpeech.LANG_NOT_SUPPORTED
    }

    override fun onGetLanguage(): Array<String> {
        return arrayOf("mya", "MMR", "")
    }

    override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int {
        return onIsLanguageAvailable(lang, country, variant)
    }

    override fun onStop() {
        // အသုံးပြုသူက ဖတ်နေတာကို ရပ်လိုက်လျှင် လုပ်ဆောင်ရန်
    }

    // --- အဓိက အလုပ်လုပ်မည့် စနစ်ကြီး ---
    override fun onSynthesizeText(request: SynthesisRequest?, callback: SynthesisCallback?) {
        if (request == null || callback == null) return
        val env = ortEnv
        if (ortSessionMm == null || env == null) {
            callback.error()
            return
        }

        val rawText = request.charSequenceText.toString().trim()
        if (rawText.isEmpty()) {
            callback.done()
            return
        }

        // ၁။ System နှင့် အသုံးပြုသူ ညှိထားသော Pitch တန်ဖိုးများကို ယူခြင်း
        val systemPitch = request.pitch
        val sharedPref = getSharedPreferences("LoTaYaSettings", Context.MODE_PRIVATE)
        val userCustomPitch = sharedPref.getFloat("custom_pitch", 1.0f)
        val finalPitch = (systemPitch.toFloat() / 100.0f) * userCustomPitch

        // ၂။ စနစ်အား အော်ဒီယို စတင်ထုတ်လုပ်မည်ဖြစ်ကြောင်း အကြောင်းကြားခြင်း (16kHz, 16-bit PCM, Mono)
        callback.start(16000, AudioFormat.ENCODING_PCM_16BIT, 1)

        try {
            // ၃။ စာသားထဲတွင် မြန်မာ/အင်္ဂလိပ် ရောနေပါက အပိုင်းပိုင်း ခွဲထုတ်ခြင်း
            val textSegments = splitByLanguage(rawText)

            for (segment in textSegments) {
                if (segment.isEnglish) {
                    // အင်္ဂလိပ်စာဖြစ်ပါက Google TTS Fallback ဖြင့် အော်ဒီယိုလှမ်းထုတ်သည်
                    if (isGoogleTtsReady) {
                        val engFloats = generateEnglishAudioFloats(segment.text)
                        if (engFloats != null && engFloats.isNotEmpty()) {
                            streamFloatsToCallback(engFloats, callback)
                        }
                    }
                } else {
                    // မြန်မာစာဖြစ်ပါက ကိုယ်ပိုင် ONNX Model ထဲ ထည့်မောင်းသည် (Pitch Control ပါဝင်မည်)
                    val mmFloats = runMyanmarPipelineWithPitch(segment.text, env, finalPitch)
                    if (mmFloats != null && mmFloats.isNotEmpty()) {
                        streamFloatsToCallback(mmFloats, callback)
                    }
                }
            }
            // ၄။ စာအားလုံးဖတ်ပြီးမြောက်ကြောင်း စနစ်ထံ အသိပေးခြင်း
            callback.done()

        } catch (e: Exception) {
            e.printStackTrace()
            callback.error()
        }
    }

    private fun runMyanmarPipelineWithPitch(text: String, env: OrtEnvironment, pitch: Float): FloatArray? {
        var processedText = preProcessMyanmarText(text)
        processedText = normalizeNumbers(processedText)

        val cleanChunk = processedText.replace(" ", "")
        val validChars = cleanChunk.filter { vocabMapMm.containsKey(it) }
        if (validChars.length < 2) return null

        val tokenList = mutableListOf<Long>()
        tokenList.add(0L)
        for (i in validChars.indices) {
            val id = vocabMapMm[validChars[i]] ?: 56L
            if (id in 0L..56L) {
                tokenList.add(id)
                tokenList.add(0L)
            }
        }

        val inputSequence = tokenList.toLongArray()
        if (inputSequence.isEmpty()) return null

        val inputShape = longArrayOf(1, inputSequence.size.toLong())
        val singleShape = longArrayOf(1)

        val inputTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(inputSequence), inputShape)
        val lengthTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(inputSequence.size.toLong())), singleShape)
        
        // 💡 ဤနေရာတွင် pitch တန်ဖိုးအား သက်ရောက်စေပြီး ကုလားသံဝဲခြင်းကို ထိန်းချုပ်ထားပါသည်
        // VITS scale array: [noise_scale, length_scale, noise_scale_w]
        // noise_scale_w (တတိယမြောက်ကောင်) ကို ပေးလိုက်တဲ့ pitch အတိုင်း မြှင့်ပေးခြင်းဖြင့် လေသံဝဲတာ သက်သာစေသည်
        val scalesTensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(floatArrayOf(0.667f, 1.0f, 0.8f * pitch)), longArrayOf(3))

        val attentionMaskSequence = LongArray(inputSequence.size) { 1L }
        val maskTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(attentionMaskSequence), inputShape)
        val sidTensor = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(0L)), singleShape)

        val inputMap = HashMap<String, OnnxTensor>()
        ortSessionMm?.inputNames?.forEach { name ->
            val lowerName = name.lowercase()
            when {
                lowerName.contains("mask") || lowerName.contains("attention") -> inputMap[name] = maskTensor
                name == "input" || lowerName.contains("input_ids") || lowerName == "text" -> inputMap[name] = inputTensor
                lowerName.contains("length") -> inputMap[name] = lengthTensor
                lowerName.contains("scale") -> inputMap[name] = scalesTensor
                lowerName.contains("sid") || lowerName.contains("speaker") -> inputMap[name] = sidTensor
            }
        }

        val results = ortSessionMm?.run(inputMap)
        val outputTensor = results?.get(0) as? OnnxTensor
        var audioFloats: FloatArray? = null
        
        outputTensor?.let {
            val floatBuffer = it.floatBuffer
            audioFloats = FloatArray(floatBuffer.remaining())
            floatBuffer.get(audioFloats)
        }

        inputTensor.close()
        lengthTensor.close()
        scalesTensor.close()
        maskTensor.close()
        sidTensor.close()
        results?.close()

        return audioFloats
    }

    // --- Float Array အား System မှ နားလည်သော 16-Bit PCM Byte အဖြစ်ပြောင်း၍ တိုက်ရိုက်ပေးပို့ခြင်း ---
    private fun streamFloatsToCallback(floatData: FloatArray, callback: SynthesisCallback) {
        val byteBuffer = ByteBuffer.allocate(floatData.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (f in floatData) {
            var s = (f * 32300.0f).toInt() // Gain အနည်းငယ်ထိန်းညှိထားသည်
            if (s > 32767) s = 32767
            if (s < -32768) s = -32768
            byteBuffer.putShort(s.toShort())
        }

        val pcmBytes = byteBuffer.array()
        val maxBufferSize = callback.maxBufferSize
        var offset = 0
        while (offset < pcmBytes.size) {
            val bytesToWrite = minOf(maxBufferSize, pcmBytes.size - offset)
            callback.audioAvailable(pcmBytes, offset, bytesToWrite)
            offset += bytesToWrite
        }
    }

    // --- ခင်ဗျားရေးထားသော မူရင်း Helper Functions များကို Service သို့ ပြောင်းရွှေ့ခြင်း ---
    data class LangSegment(val text: String, val isEnglish: Boolean)

    private fun splitByLanguage(text: String): List<LangSegment> {
        val segments = mutableListOf<LangSegment>()
        if (text.isEmpty()) return segments
        var currentSegment = StringBuilder()
        var currentIsEnglish: Boolean? = null

        for (ch in text) {
            val computedIsEnglish = if (ch in " .,!?'-" && currentIsEnglish != null) {
                currentIsEnglish!!
            } else {
                ch in 'a'..'z' || ch in 'A'..'Z'
            }

            if (currentIsEnglish == null) {
                currentIsEnglish = computedIsEnglish
                currentSegment.append(ch)
            } else if (currentIsEnglish == computedIsEnglish) {
                currentSegment.append(ch)
            } else {
                if (currentSegment.toString().trim().isNotEmpty()) {
                    segments.add(LangSegment(currentSegment.toString(), currentIsEnglish!!))
                }
                currentSegment = StringBuilder().append(ch)
                currentIsEnglish = computedIsEnglish
            }
        }
        if (currentSegment.toString().trim().isNotEmpty()) {
            segments.add(LangSegment(currentSegment.toString(), currentIsEnglish ?: false))
        }
        return segments
    }

    private fun generateEnglishAudioFloats(text: String): FloatArray? {
        val tempWaveFile = File(cacheDir, "temp_eng_${UUID.randomUUID()}.wav")
        val latch = CountDownLatch(1)

        googleTts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onDone(utteranceId: String?) { latch.countDown() }
            override fun onError(utteranceId: String?) { latch.countDown() }
        })

        val params = Bundle()
        params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "EngWave")
        googleTts?.synthesizeToFile(text, params, tempWaveFile, "EngWave")

        try {
            latch.await(10, TimeUnit.SECONDS)
            if (tempWaveFile.exists() && tempWaveFile.length() > 44) {
                val bytes = tempWaveFile.readBytes()
                tempWaveFile.delete()

                val pcmByteSize = bytes.size - 44
                val shortSamples = pcmByteSize / 2
                val floatArray = FloatArray(shortSamples)

                val buffer = ByteBuffer.wrap(bytes, 44, pcmByteSize).order(ByteOrder.LITTLE_ENDIAN)
                for (i in 0 until shortSamples) {
                    if (buffer.hasRemaining()) {
                        floatArray[i] = buffer.short / 32768.0f
                    }
                }
                return downsampleTo16k(floatArray, 24000)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return null
    }

    private fun downsampleTo16k(input: FloatArray, srcSampleRate: Int): FloatArray {
        if (srcSampleRate == 16000) return input
        val ratio = srcSampleRate.toDouble() / 16000.0
        val destLength = (input.size / ratio).toInt()
        val output = FloatArray(destLength)
        for (i in 0 until destLength) {
            val srcIndex = (i * ratio).toInt()
            if (srcIndex < input.size) {
                output[i] = input[srcIndex]
            }
        }
        return output
    }

    private fun preProcessMyanmarText(text: String): String {
        var res = text
        res = res.replace(Regex("[xX._*#+=()_\\-]"), "")
        res = res.replace("ဪ", "အော်")
        res = res.replace("၎င်း", "လဂေါင်း")
        res = res.replace("ဖြစ်၏", "ဖြစ်အီ")
        res = res.replace("ဗျ", "ဗျ")
        res = res.replace("၏", "အီး")
        res = res.replace("၌", "နှိုက်")
        res = res.replace("၍", "ရွေ့")
        res = res.replace("ဤ", "အီ")
        res = res.replace("ဘဏ္ဍာ", "ဘန်ဒါ")
        res = res.replace("သဏ္ဌာန်", "သန်ထန်")
        if (!res.endsWith(" ")) res = "$res   "
        return res
    }

    private fun normalizeNumbers(text: String): String {
        var res = text
        val numMap = mapOf(
            "0" to "သုည", "1" to "တစ်", "2" to "နှစ်", "3" to "သုံး", "4" to "လေး", "5" to "ငါး",
            "6" to "ခြောက်", "7" to "ခုနစ်", "8" to "ရှစ်", "9" to "ကိုး",
            "၀" to "သုည", "၁" to "တစ်", "၂" to "နှစ်", "၃" to "သုံး", "၄" to "လေး", "၅" to "ငါး",
            "၆" to "ခြောက်", "၇" to "ခုနစ်", "၈" to "ရှစ်", "၉" to "ကိုး"
        )
        for ((num, txt) in numMap) { res = res.replace(num, txt) }
        return res
    }

    private fun copyAssetToFile(assetName: String, outFile: File) {
        assets.open(assetName).use { inputStream ->
            FileOutputStream(outFile).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        googleTts?.stop()
        googleTts?.shutdown()
        ortSessionMm?.close()
        ortEnv?.close()
    }
}
