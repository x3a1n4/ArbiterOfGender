import discord
from discord.ext import commands, voice_recv
from transformers import pipeline
import asyncio
import io
import numpy as np
import logging
# logging.getLogger("discord.ext.voice_recv.router").setLevel(logging.CRITICAL)
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.CRITICAL)
logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Crash handling
# ---------------------------------------------------------------------------
import discord.opus
from discord.ext.voice_recv import opus as vr_opus

_original_process_packet = vr_opus.PacketDecoder._process_packet

def _safe_process_packet(self, packet):
    try:
        return _original_process_packet(self, packet)
    except discord.opus.OpusError:
        return None

vr_opus.PacketDecoder._process_packet = _safe_process_packet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLIP_SECONDS = 2          # analyse every N seconds of speech per user
SAMPLE_RATE  = 48000      # Discord PCM: 48 kHz, stereo, 16-bit
CHANNELS     = 2
BYTES_PER_SAMPLE = 2      # int16
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE   # 192 000
CLIP_BYTES   = BYTES_PER_SECOND * CLIP_SECONDS                  # 960 000

# ---------------------------------------------------------------------------
# AI model
# ---------------------------------------------------------------------------
classifier = pipeline(
    "audio-classification",
    model="prithivMLmods/Common-Voice-Gender-Detection"
)

# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True

APP_ID = 1475022407071830124
bot = commands.Bot(command_prefix="!", intents=intents, application_id=APP_ID)


@bot.command()
@commands.is_owner()
async def sync(ctx):
    bot.tree.copy_global_to(guild=ctx.guild)
    synced = await bot.tree.sync(guild=ctx.guild)
    await ctx.send(f"Synced {len(synced)} commands.")


@bot.tree.command(name="join", description="Join a voice channel and moderate by gender")
@commands.has_role("Mod")
async def join_vc(interaction: discord.Interaction, channel: discord.VoiceChannel):
    vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
    # await interaction.response.send_message(f"Joined **{channel.name}** — monitoring started.")
    print(f"Joined **{channel.name}** — monitoring started.")

    # Pass the text channel where the command was used for status messages
    sink = GenderSink(
        voice_channel=channel,
        text_channel=interaction.channel,
        voice_client=vc,
    )
    vc.listen(sink)
    
    await interaction.response.send_message(f"Joined **{channel.name}**")

@bot.tree.command(name="leave", description="Leave the voice channel")
@commands.has_role("Mod")
async def leave_vc(interaction: discord.Interaction):
    vc = interaction.guild.voice_client
    if vc is None:
        await interaction.response.send_message("I'm not in a voice channel.")
        return
    await vc.disconnect()
    await interaction.response.send_message("Disconnected.")

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------
class GenderSink(voice_recv.AudioSink):
    """
    Buffers PCM audio per user. Every time a user accumulates CLIP_BYTES
    of audio data, their clip is analysed. If the top label is 'male',
    they are kicked from the voice channel.
    """

    def __init__(
        self,
        voice_channel: discord.VoiceChannel,
        text_channel: discord.TextChannel,
        voice_client: voice_recv.VoiceRecvClient,
    ):
        super().__init__()
        self.voice_channel = voice_channel
        self.text_channel  = text_channel
        self.vc  = voice_client

        # user_id -> BytesIO buffer
        self.buffers: dict[int, io.BytesIO] = {}
        self.pending_kick: set[int] = set()  # users currently being kicked

    # ------------------------------------------------------------------
    # AudioSink interface
    # ------------------------------------------------------------------

    def wants_opus(self) -> bool:
        return False   # we want decoded PCM

    def write(self, user: discord.Member, data: voice_recv.VoiceData):
        if user is None or user.bot:
            return
        
        if user.id in self.pending_kick:
            return
        

        uid = user.id
        if uid not in self.buffers:
            self.buffers[uid] = io.BytesIO()

        buf = self.buffers[uid]
        buf.write(data.pcm)

        if buf.tell() >= CLIP_BYTES:
            raw = buf.getvalue()
            self.buffers[uid] = io.BytesIO()
            
            # Run classifier synchronously right here on the router thread
            # This avoids all executor/event loop threading issues
            try:
                label, score = self._run_classifier(raw)
                print(f"User {user.display_name}: {label} ({score:.1%})")
            except Exception as e:
                print(f"Classifier error: {e}")
                return
            
            if label.lower() == "male":
                self.pending_kick.add(uid)  # stop processing their packets immediately

            # Only the Discord API calls need the event loop
            asyncio.run_coroutine_threadsafe(
                self._act(user, label, score),
                bot.loop
            )

    async def _act(self, user: discord.Member, label: str, score: float):
        """
        await self.text_channel.send(
            f"**{user.display_name}** — `{label}` ({score:.1%} confidence)"
        )
        """
        if label.lower() == "male":
            await self.text_channel.send(
                f"**{user.display_name}** was removed ({score:.2%} confidence)"
            )
            await self._kick(user)

    def on_rtcp_packet(self, packet, guild):
        pass   # suppress warnings

    def cleanup(self):
        for buf in self.buffers.values():
            buf.close()
        self.buffers.clear()

    # ------------------------------------------------------------------
    # Analysis + moderation
    # ------------------------------------------------------------------
    def _run_classifier(self, raw: bytes) -> tuple[str, float]:
        audio_np = (
            np.frombuffer(raw, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )
        # Stereo → mono by averaging channels
        audio_mono = audio_np.reshape(-1, CHANNELS).mean(axis=1)
        
        result = classifier({"raw": audio_mono, "sampling_rate": SAMPLE_RATE})
        top = result[0]
        return top["label"], top["score"]

    async def _kick(self, user: discord.Member):
        try:
            await user.move_to(None)   # disconnects from voice
            self.pending_kick.remove(user.id) # add back to list
        except discord.Forbidden:
            print(
                f"No permission to move **{user.display_name}**."
            )
        except discord.HTTPException as e:
            print(f"Could not remove {user.display_name}: {e}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
with open("token.txt") as f:
    token = f.read().strip()

bot.run(token)