IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[AuditJobTierCategory]') AND type in (N'U'))
DROP TABLE [dbo].[AuditJobTierCategory]
GO
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

SET ANSI_PADDING ON
GO

CREATE TABLE [dbo].[AuditJobTierCategory](
	[JobCode] [varchar](20) NOT NULL,
	[TierId] [int] NOT NULL,
	[CategoryCode] [varchar](10) NOT NULL,
	[CategoryValue] [varchar](25) NOT NULL,
	[Active] [bit] NOT NULL,
	[LastUser] [int] NOT NULL,
	[LastUpdate] [datetime] NOT NULL,
	[RecordInfo] [varchar](25) NOT NULL,
	[UpdatedDatetime] [datetime] NOT NULL,
	[UpdatedUser] [varchar](50) NOT NULL
) ON [PRIMARY]

GO

SET ANSI_PADDING OFF
GO
IF  EXISTS (SELECT * FROM sys.indexes WHERE object_id = OBJECT_ID(N'[dbo].[AuditJobTierCategory]') AND name = N'CIX_AuditJobTierCategory')
DROP INDEX [CIX_AuditJobTierCategory] ON [dbo].[AuditJobTierCategory] WITH ( ONLINE = OFF )
GO
CREATE CLUSTERED INDEX [CIX_AuditJobTierCategory] ON [dbo].[AuditJobTierCategory] 
(
	[UpdatedDatetime] ASC,
	[JobCode] ASC,
	[TierId] ASC,
	[CategoryCode] ASC,
	[RecordInfo] ASC
)WITH (PAD_INDEX  = OFF, STATISTICS_NORECOMPUTE  = OFF, SORT_IN_TEMPDB = OFF, IGNORE_DUP_KEY = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS  = ON, ALLOW_PAGE_LOCKS  = ON) ON [PRIMARY]
GO


